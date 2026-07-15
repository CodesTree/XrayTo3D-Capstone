Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function New-MarkdownCell([string]$Source) {
    [ordered]@{
        cell_type = 'markdown'
        metadata = [ordered]@{}
        source = @($Source -split "(?<=`n)")
    }
}

function New-CodeCell([string]$Source, [string[]]$Tags = @()) {
    $metadata = [ordered]@{}
    if ($Tags.Count -gt 0) { $metadata.tags = $Tags }
    [ordered]@{
        cell_type = 'code'
        execution_count = $null
        metadata = $metadata
        outputs = @()
        source = @($Source -split "(?<=`n)")
    }
}

function Write-Notebook([string]$Path, [object[]]$Cells) {
    $nb = [ordered]@{
        cells = $Cells
        metadata = [ordered]@{
            kernelspec = [ordered]@{ display_name = 'Python 3'; language = 'python'; name = 'python3' }
            language_info = [ordered]@{ name = 'python'; version = '3.12' }
        }
        nbformat = 4
        nbformat_minor = 5
    }
    $json = $nb | ConvertTo-Json -Depth 100
    $full = Join-Path $PSScriptRoot $Path
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($full)) | Out-Null
    [IO.File]::WriteAllText($full, $json + "`n", [Text.UTF8Encoding]::new($false))
}

$nb01 = @()
$nb01 += New-MarkdownCell @'
# Stage 2.1 — Fold-isolated FCMAE P1 and bidirectional cross-view P2

This notebook implements the authoritative `baseline_protocol_v1` encoder foundation.

- It consumes only the certified 71-knee manifest and never opens test-fold files.
- P1 uses canonical uniform 60% FCMAE masking; no target, fracture label, cohort sampler, or structure-derived mask enters pretraining.
- P2 adds bidirectional AP/LAT completion while retaining the FCMAE loss.
- Training augmentation is deterministic and photometric only. Validation is always clean.
- Every fold starts independently from the pinned self-supervised `convnextv2_tiny.fcmae` initialization.

`RUN_REAL_DATA` is deliberately `False` by default. A real run must be started only from an approved Stage 2 task brief after Agent N records Stage 1 `PASS`.
'@

$nb01 += New-CodeCell @'
import contextlib
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, get_worker_info
import timm

# Environment/configuration cell. Algorithm cells are identical in the local and HPC twins.
PROTOCOL_VERSION = "baseline_protocol_v1"
STAGE2_SCHEMA = "foundation_stage2_v1"
SEED = 42
FOLD = 0
RUN_REAL_DATA = False
RUN_P1 = False
RUN_P2 = False
RESUME_P1 = None
RESUME_P2 = None

PRETRAIN_MODEL = "convnextv2_tiny.fcmae"
IMAGE_SIZE = 256
PATCH_SIZE = 32
MASK_RATIO = 0.60
DECODER_DIM = 512
XVIEW_WEIGHT = 1.0
XVIEW_ROW_SLACK = 1

P1_MAX_EPOCHS = 250
P2_MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 30
MICRO_BATCH = 16
ACCUM_STEPS = 8
EFFECTIVE_BATCH = 128
BASE_LR = 1.5e-4
PEAK_LR = BASE_LR * EFFECTIVE_BATCH / 256
WARMUP_FRACTION = 0.08
WEIGHT_DECAY = 0.05
NUM_WORKERS = 4 if os.name != "nt" else 0
USE_AMP = True
CHECKPOINT_EVERY = 25

AUGMENTATION = {
    "kind": "online_photometric_only",
    "gamma": [0.90, 1.10],
    "brightness": [-0.05, 0.05],
    "gaussian_noise_sigma": [0.0, 0.02],
    "clamp": [0.0, 1.0],
    "independent_ap_lat": True,
    "forbidden": ["crop", "rotation", "translation", "flip", "elastic", "cutout", "random_erasing"],
}

assert 0 <= FOLD < 5
assert MICRO_BATCH * ACCUM_STEPS == EFFECTIVE_BATCH
assert IMAGE_SIZE % PATCH_SIZE == 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print({"fold": FOLD, "device": str(DEVICE), "run_real_data": RUN_REAL_DATA, "peak_lr": PEAK_LR})
'@ @('environment-configuration')

$nb01 += New-CodeCell @'
def find_project_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "configs" / "baseline_protocol_v1.json").exists() and (candidate / "reports" / "manifests").exists():
            return candidate
    raise FileNotFoundError("Project root with configs/ and reports/manifests/ was not found")


ROOT = find_project_root(Path.cwd())
MANIFEST_PATH = ROOT / "reports" / "manifests" / "quantitative_manifest_v1.csv"
MANIFEST_META_PATH = ROOT / "reports" / "manifests" / "quantitative_manifest_v1.metadata.json"
BASELINE_CONFIG_PATH = ROOT / "configs" / "baseline_protocol_v1.json"
DATA_CONFIG_PATH = ROOT / "configs" / "data_contract_v1.json"
ARTIFACT_ROOT = ROOT / "models" / STAGE2_SCHEMA / f"fold_{FOLD}"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def tensor_state_sha256(state: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(value.dtype).encode("ascii"))
        h.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def stable_seed(*parts) -> int:
    token = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def augment_drr(array: np.ndarray, sample_id: str, view: str, fold: int, epoch: int, worker_id: int) -> np.ndarray:
    """Deterministic training-only photometric augmentation; geometry is never changed."""
    rng = np.random.default_rng(stable_seed(SEED, fold, epoch, worker_id, sample_id, view))
    gamma = rng.uniform(*AUGMENTATION["gamma"])
    brightness = rng.uniform(*AUGMENTATION["brightness"])
    sigma = rng.uniform(*AUGMENTATION["gaussian_noise_sigma"])
    out = np.power(np.clip(array, 0.0, 1.0), gamma, dtype=np.float32)
    out = out + np.float32(brightness)
    if sigma > 0:
        out = out + rng.normal(0.0, sigma, size=out.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def load_certified_folds(fold: int):
    baseline = json.loads(BASELINE_CONFIG_PATH.read_text(encoding="utf-8"))
    data_contract = json.loads(DATA_CONFIG_PATH.read_text(encoding="utf-8"))
    meta = json.loads(MANIFEST_META_PATH.read_text(encoding="utf-8"))
    if baseline["protocol_version"] != PROTOCOL_VERSION:
        raise RuntimeError("baseline protocol version mismatch")
    if not meta.get("certification_approved", False):
        raise RuntimeError("Stage 1 is not certified: certification_approved is false")
    if int(meta.get("ready_rows", -1)) != 71 or int(meta.get("pending_recertification_rows", -1)) != 0:
        raise RuntimeError("Stage 1 must contain exactly 71 ready rows and zero pending rows")
    actual_manifest_sha = sha256_file(MANIFEST_PATH)
    if actual_manifest_sha != meta.get("sha256"):
        raise RuntimeError("manifest SHA-256 does not match quantitative_manifest_v1.metadata.json")

    all_rows = pd.read_csv(MANIFEST_PATH, dtype={"test_fold": "Int64"})
    ready = all_rows[all_rows["status"].eq("ready")].copy()
    if len(ready) != 71:
        raise RuntimeError(f"expected 71 ready manifest rows, found {len(ready)}")
    counts = ready.groupby("dataset").size().to_dict()
    if counts != {"Ruikar": 13, "VSD": 58}:
        raise RuntimeError(f"cohort mismatch: {counts}")
    if set(ready["test_fold"].astype(int)) != set(range(5)):
        raise RuntimeError("test_fold must contain only 0..4")
    if ready.groupby("subject_id")["test_fold"].nunique().max() != 1:
        raise RuntimeError("subject leakage across test folds")
    if ready["orientation"].ne("LPS").any():
        raise RuntimeError("non-LPS row in certified manifest")
    if ready["target_version"].ne(data_contract["target_version"]).any():
        raise RuntimeError("target version mismatch")
    if ready["drr_version"].ne(data_contract["drr_version"]).any():
        raise RuntimeError("DRR version mismatch")

    ready["test_fold"] = ready["test_fold"].astype(int)
    ready["split"] = "train"
    ready.loc[ready["test_fold"].eq(fold), "split"] = "test"
    ready.loc[ready["test_fold"].eq((fold + 1) % 5), "split"] = "validation"
    split_subjects = {name: set(group["subject_id"]) for name, group in ready.groupby("split")}
    if split_subjects["train"] & split_subjects["validation"] or split_subjects["train"] & split_subjects["test"] or split_subjects["validation"] & split_subjects["test"]:
        raise RuntimeError("subject overlap between train/validation/test")

    # Only train and validation paths are checked/opened in Stage 2. Test paths remain unopened.
    for split in ("train", "validation"):
        for row in ready[ready["split"].eq(split)].itertuples(index=False):
            for field in ("ap_drr_path", "lat_drr_path"):
                path = ROOT / getattr(row, field)
                if not path.is_file():
                    raise FileNotFoundError(f"missing certified {split} input: {path}")

    return ready, ready[ready.split.eq("train")].copy(), ready[ready.split.eq("validation")].copy(), ready[ready.split.eq("test")].copy(), meta


def read_clean_drr(path: Path) -> np.ndarray:
    array = np.load(path).astype(np.float32)
    if array.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"DRR shape must be 256x256, got {array.shape} for {path}")
    if not np.isfinite(array).all() or array.min() < -1e-6 or array.max() > 1.0 + 1e-6:
        raise ValueError(f"DRR must be finite and within [0,1]: {path}")
    return np.clip(array, 0.0, 1.0)


class AnchorDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, training: bool):
        self.training = training
        self.epoch = 0
        self.records = []
        for row in rows.itertuples(index=False):
            self.records.extend([
                (row.sample_id, "ap", ROOT / row.ap_drr_path),
                (row.sample_id, "lat", ROOT / row.lat_drr_path),
            ])

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        sample_id, view, path = self.records[index]
        array = read_clean_drr(path)
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        if self.training:
            array = augment_drr(array, sample_id, view, FOLD, self.epoch, worker_id)
        return {"image": torch.from_numpy(array).unsqueeze(0), "sample_id": sample_id, "view": view}


class PairDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, training: bool):
        self.rows = rows.reset_index(drop=True)
        self.training = training
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        ap = read_clean_drr(ROOT / row.ap_drr_path)
        lat = read_clean_drr(ROOT / row.lat_drr_path)
        if self.training:
            ap = augment_drr(ap, row.sample_id, "ap", FOLD, self.epoch, worker_id)
            lat = augment_drr(lat, row.sample_id, "lat", FOLD, self.epoch, worker_id)
        return {"ap": torch.from_numpy(ap).unsqueeze(0), "lat": torch.from_numpy(lat).unsqueeze(0), "sample_id": row.sample_id}


seed_everything()
print("project root:", ROOT)
'@

$nb01 += New-CodeCell @'
MASK_GRID = IMAGE_SIZE // PATCH_SIZE


def build_backbone(pretrained: bool):
    """No fallback is permitted: the pinned FCMAE model must load or the run fails."""
    return timm.create_model(PRETRAIN_MODEL, pretrained=pretrained, features_only=True)


class FCMAEEncoder(nn.Module):
    """Dense-mask equivalent of the sparse FCMAE encoder, with re-zeroing after every block."""
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        required = ["stem_0", "stem_1", *[f"stages_{i}" for i in range(4)]]
        missing = [name for name in required if not hasattr(backbone, name)]
        if missing:
            raise RuntimeError(f"unsupported timm features_only wrapper; missing {missing}")

    @staticmethod
    def _visible(x, mask):
        visible = F.interpolate((~mask).float(), size=x.shape[-2:], mode="nearest")
        return x * visible

    def forward(self, x, mask):
        bb = self.backbone
        x = self._visible(x, mask)
        x = bb.stem_1(bb.stem_0(x))
        x = self._visible(x, mask)
        features = []
        for index in range(4):
            stage = getattr(bb, f"stages_{index}")
            x = stage.downsample(x)
            x = self._visible(x, mask)
            for block in stage.blocks:
                x = block(x)
                x = self._visible(x, mask)
            features.append(x)
        return features


class ConvNeXtDecoderBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.pointwise1 = nn.Conv2d(channels, 4 * channels, 1)
        self.pointwise2 = nn.Conv2d(4 * channels, channels, 1)

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.pointwise2(F.gelu(self.pointwise1(x)))
        return x + residual


class FCMAEDecoder(nn.Module):
    def __init__(self, input_dim=768, decoder_dim=DECODER_DIM):
        super().__init__()
        self.projection = nn.Conv2d(input_dim, decoder_dim, 1)
        self.block = ConvNeXtDecoderBlock(decoder_dim)
        self.prediction = nn.Conv2d(decoder_dim, 3 * PATCH_SIZE * PATCH_SIZE, 1)

    def forward(self, feature):
        pixels = self.prediction(self.block(self.projection(feature)))
        batch = pixels.shape[0]
        pixels = pixels.view(batch, 3, PATCH_SIZE, PATCH_SIZE, MASK_GRID, MASK_GRID)
        return pixels.permute(0, 1, 4, 5, 2, 3).contiguous()


class CrossViewBlock(nn.Module):
    def __init__(self, dim=768, heads=8):
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, target, source, attention_mask):
        query = self.self_norm(target)
        target = target + self.self_attention(query, query, query, need_weights=False)[0]
        query = self.cross_norm(target)
        target = target + self.cross_attention(query, source, source, attn_mask=attention_mask, need_weights=False)[0]
        return target + self.mlp(self.mlp_norm(target))


class CrossViewDecoder(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.block = CrossViewBlock(dim)
        self.prediction = nn.Linear(dim, 3 * PATCH_SIZE * PATCH_SIZE)

    def forward(self, target_feature, source_feature, attention_mask):
        batch, channels, height, width = target_feature.shape
        target = target_feature.flatten(2).transpose(1, 2)
        source = source_feature.flatten(2).transpose(1, 2)
        pixels = self.prediction(self.block(target, source, attention_mask))
        pixels = pixels.view(batch, height, width, 3, PATCH_SIZE, PATCH_SIZE)
        return pixels.permute(0, 3, 1, 2, 4, 5).contiguous()


def rowwise_attention_mask(device):
    index = torch.arange(MASK_GRID * MASK_GRID, device=device)
    rows = index // MASK_GRID
    mask = torch.zeros((index.numel(), index.numel()), device=device)
    mask[(rows[:, None] - rows[None, :]).abs() > XVIEW_ROW_SLACK] = float("-inf")
    return mask


def uniform_mask(batch_size: int, device, generator: torch.Generator):
    count = MASK_GRID * MASK_GRID
    masked = int(round(MASK_RATIO * count))
    noise = torch.rand((batch_size, count), generator=generator, device="cpu")
    selected = noise.argsort(dim=1)[:, :masked]
    mask = torch.zeros((batch_size, count), dtype=torch.bool)
    mask.scatter_(1, selected, True)
    return mask.view(batch_size, 1, MASK_GRID, MASK_GRID).to(device)


def patch_normalized_target(raw_one_channel):
    raw_three_channel = raw_one_channel.repeat(1, 3, 1, 1)
    patches = raw_three_channel.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
    mean = patches.mean(dim=(-1, -2), keepdim=True)
    variance = patches.var(dim=(-1, -2), keepdim=True, unbiased=False)
    return (patches - mean) / torch.sqrt(variance + 1e-6)


def masked_mse(prediction, target, mask):
    weights = mask.unsqueeze(-1).unsqueeze(-1).float()
    return (((prediction - target) ** 2) * weights).sum() / (weights.sum() * prediction.shape[1] * PATCH_SIZE * PATCH_SIZE).clamp_min(1.0)


def encoder_input(raw, mean, std):
    image = raw.repeat(1, 3, 1, 1)
    mean_tensor = torch.as_tensor(mean, dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
    std_tensor = torch.as_tensor(std, dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
    return (image - mean_tensor) / std_tensor


def p1_forward(encoder, decoder, raw, mean, std, generator):
    raw = raw.to(DEVICE, non_blocking=True)
    mask = uniform_mask(raw.shape[0], DEVICE, generator)
    feature = encoder(encoder_input(raw, mean, std), mask)[-1]
    prediction = decoder(feature)
    loss = masked_mse(prediction, patch_normalized_target(raw), mask)
    return loss, feature


def p2_forward(encoder, fcmae_decoder, cross_decoder, ap, lat, mean, std, generator):
    ap = ap.to(DEVICE, non_blocking=True)
    lat = lat.to(DEVICE, non_blocking=True)
    ap_mask = uniform_mask(ap.shape[0], DEVICE, generator)
    lat_mask = uniform_mask(lat.shape[0], DEVICE, generator)
    ap_feature = encoder(encoder_input(ap, mean, std), ap_mask)[-1]
    lat_feature = encoder(encoder_input(lat, mean, std), lat_mask)[-1]
    ap_target = patch_normalized_target(ap)
    lat_target = patch_normalized_target(lat)
    fcmae = 0.5 * (
        masked_mse(fcmae_decoder(ap_feature), ap_target, ap_mask)
        + masked_mse(fcmae_decoder(lat_feature), lat_target, lat_mask)
    )
    attention_mask = rowwise_attention_mask(DEVICE)
    ap_from_lat = masked_mse(cross_decoder(ap_feature, lat_feature, attention_mask), ap_target, ap_mask)
    lat_from_ap = masked_mse(cross_decoder(lat_feature, ap_feature, attention_mask), lat_target, lat_mask)
    cross = 0.5 * (ap_from_lat + lat_from_ap)
    return fcmae + XVIEW_WEIGHT * cross, {"fcmae": fcmae.detach(), "cross": cross.detach()}, (ap_feature, lat_feature)
'@

$nb01 += New-CodeCell @'
def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def amp_context():
    if not (USE_AMP and DEVICE.type == "cuda"):
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_scaler():
    enabled = USE_AMP and DEVICE.type == "cuda" and not torch.cuda.is_bf16_supported()
    return torch.amp.GradScaler("cuda", enabled=enabled)


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def scheduler_for(optimizer, total_updates: int):
    warmup = max(1, int(round(WARMUP_FRACTION * total_updates)))
    def multiplier(step):
        if step < warmup:
            return max(1e-8, (step + 1) / warmup)
        progress = (step - warmup) / max(1, total_updates - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def save_training_state(path, stage, epoch, global_step, modules, optimizer, scheduler, scaler, best_metric, config_sha):
    payload = {
        "schema_version": STAGE2_SCHEMA,
        "stage": stage,
        "fold": FOLD,
        "epoch": epoch,
        "global_step": global_step,
        "modules": {name: module.state_dict() for name, module in modules.items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_state": rng_state(),
        "best_metric": float(best_metric),
        "config_sha256": config_sha,
    }
    torch.save(payload, path)


def strict_load(module, state, label):
    incompatible = module.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"{label} strict-load failure: {incompatible}")


def run_config(stage, manifest_meta, train_rows, validation_rows, test_rows, upstream_sha=None):
    return {
        "schema_version": STAGE2_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "stage": stage,
        "fold": FOLD,
        "seed": SEED,
        "manifest_sha256": manifest_meta["sha256"],
        "baseline_protocol_sha256": sha256_file(BASELINE_CONFIG_PATH),
        "data_contract_sha256": sha256_file(DATA_CONFIG_PATH),
        "train_sample_ids": sorted(train_rows.sample_id.tolist()),
        "validation_sample_ids": sorted(validation_rows.sample_id.tolist()),
        "test_sample_ids_not_opened": sorted(test_rows.sample_id.tolist()),
        "pretrained_model": PRETRAIN_MODEL,
        "upstream_checkpoint_sha256": upstream_sha,
        "augmentation": AUGMENTATION,
        "hyperparameters": {
            "mask_ratio": MASK_RATIO,
            "patch_size": PATCH_SIZE,
            "decoder_dim": DECODER_DIM,
            "effective_batch": EFFECTIVE_BATCH,
            "micro_batch": MICRO_BATCH,
            "accum_steps": ACCUM_STEPS,
            "peak_lr": PEAK_LR,
            "weight_decay": WEIGHT_DECAY,
            "warmup_fraction": WARMUP_FRACTION,
            "p1_max_epochs": P1_MAX_EPOCHS,
            "p2_max_epochs": P2_MAX_EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "xview_weight": XVIEW_WEIGHT,
            "xview_row_slack": XVIEW_ROW_SLACK,
        },
        "software": {"python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__},
        "device": str(DEVICE),
    }


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_p1(train_rows, validation_rows, test_rows, manifest_meta):
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    config = run_config("FCMAE_P1", manifest_meta, train_rows, validation_rows, test_rows)
    config_sha = canonical_sha256(config)
    write_json(ARTIFACT_ROOT / "fcmae_p1_config.json", config)

    backbone = build_backbone(pretrained=True)
    pretrained_cfg = dict(backbone.pretrained_cfg)
    mean, std = pretrained_cfg["mean"], pretrained_cfg["std"]
    initialization_sha = tensor_state_sha256(backbone.state_dict())
    config["pretrained_configuration"] = pretrained_cfg
    config["pretrained_state_sha256"] = initialization_sha
    write_json(ARTIFACT_ROOT / "fcmae_p1_config.json", config)

    encoder = FCMAEEncoder(backbone).to(DEVICE)
    decoder = FCMAEDecoder().to(DEVICE)
    modules = {"encoder": encoder, "fcmae_decoder": decoder}
    optimizer = torch.optim.AdamW(
        [parameter for module in modules.values() for parameter in module.parameters()],
        lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY,
    )
    train_set = AnchorDataset(train_rows, training=True)
    validation_set = AnchorDataset(validation_rows, training=False)
    train_loader = DataLoader(train_set, batch_size=MICRO_BATCH, shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
    validation_loader = DataLoader(validation_set, batch_size=MICRO_BATCH, shuffle=False, num_workers=NUM_WORKERS)
    updates_per_epoch = math.ceil(len(train_loader) / ACCUM_STEPS)
    scheduler = scheduler_for(optimizer, P1_MAX_EPOCHS * updates_per_epoch)
    scaler = make_scaler()
    start_epoch, global_step, best, patience, history = 0, 0, float("inf"), 0, []

    if RESUME_P1:
        checkpoint = torch.load(RESUME_P1, map_location=DEVICE, weights_only=False)
        if checkpoint["config_sha256"] != config_sha or checkpoint["fold"] != FOLD:
            raise RuntimeError("P1 resume checkpoint config/fold mismatch")
        for name, module in modules.items(): strict_load(module, checkpoint["modules"][name], name)
        optimizer.load_state_dict(checkpoint["optimizer"]); scheduler.load_state_dict(checkpoint["scheduler"]); scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch, global_step, best = checkpoint["epoch"] + 1, checkpoint["global_step"], checkpoint["best_metric"]

    initial_validation = None
    for epoch in range(start_epoch, P1_MAX_EPOCHS):
        train_set.set_epoch(epoch); encoder.train(); decoder.train(); optimizer.zero_grad(set_to_none=True)
        train_total, train_count = 0.0, 0
        for index, batch in enumerate(train_loader):
            generator = make_generator(stable_seed(SEED, FOLD, "p1", epoch, index))
            with amp_context(): loss, _ = p1_forward(encoder, decoder, batch["image"], mean, std, generator)
            if not torch.isfinite(loss): raise FloatingPointError("non-finite P1 loss")
            scaler.scale(loss / ACCUM_STEPS).backward()
            if (index + 1) % ACCUM_STEPS == 0 or index + 1 == len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_([p for m in modules.values() for p in m.parameters()], 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step(); global_step += 1
            train_total += loss.item() * batch["image"].shape[0]; train_count += batch["image"].shape[0]

        encoder.eval(); decoder.eval(); validation_total, validation_count, feature_std = 0.0, 0, []
        with torch.no_grad():
            for index, batch in enumerate(validation_loader):
                generator = make_generator(stable_seed(SEED, FOLD, "p1_validation", index))
                with amp_context(): loss, feature = p1_forward(encoder, decoder, batch["image"], mean, std, generator)
                validation_total += loss.item() * batch["image"].shape[0]; validation_count += batch["image"].shape[0]
                feature_std.append(float(feature.float().std().item()))
        validation_loss = validation_total / validation_count
        if initial_validation is None: initial_validation = validation_loss
        record = {"epoch": epoch, "train_loss": train_total / train_count, "validation_loss": validation_loss, "lr": optimizer.param_groups[0]["lr"], "feature_std": float(np.mean(feature_std))}
        history.append(record); pd.DataFrame(history).to_csv(ARTIFACT_ROOT / "fcmae_p1_history.csv", index=False)
        save_training_state(ARTIFACT_ROOT / "fcmae_p1_last_trainstate.pth", "FCMAE_P1", epoch, global_step, modules, optimizer, scheduler, scaler, best, config_sha)
        if epoch % CHECKPOINT_EVERY == 0:
            save_training_state(ARTIFACT_ROOT / f"fcmae_p1_epoch{epoch:03d}_trainstate.pth", "FCMAE_P1", epoch, global_step, modules, optimizer, scheduler, scaler, best, config_sha)
        if validation_loss < best:
            best, patience = validation_loss, 0
            save_training_state(ARTIFACT_ROOT / "fcmae_p1_best_trainstate.pth", "FCMAE_P1", epoch, global_step, modules, optimizer, scheduler, scaler, best, config_sha)
        else:
            patience += 1
        print(record)
        if patience >= EARLY_STOP_PATIENCE: break

    best_checkpoint = torch.load(ARTIFACT_ROOT / "fcmae_p1_best_trainstate.pth", map_location="cpu", weights_only=False)
    strict_load(encoder.cpu(), best_checkpoint["modules"]["encoder"], "P1 encoder")
    if not best < initial_validation:
        raise RuntimeError(f"P1 did not improve over epoch-zero validation: initial={initial_validation}, best={best}")
    export = {"schema_version": STAGE2_SCHEMA, "stage": "FCMAE_P1", "fold": FOLD, "encoder_state": encoder.backbone.state_dict(), "config_sha256": config_sha, "manifest_sha256": manifest_meta["sha256"], "pretrained_state_sha256": initialization_sha}
    torch.save(export, ARTIFACT_ROOT / "fcmae_p1_encoder.pth")
    provenance = {**config, "config_sha256": config_sha, "best_validation_loss": best, "initial_validation_loss": initial_validation, "encoder_sha256": tensor_state_sha256(export["encoder_state"]), "trainstate_sha256": sha256_file(ARTIFACT_ROOT / "fcmae_p1_best_trainstate.pth")}
    write_json(ARTIFACT_ROOT / "fcmae_p1_provenance.json", provenance)
    return provenance


def crossview_pairing_probe(encoder, cross_decoder, validation_rows, mean, std):
    loader = DataLoader(PairDataset(validation_rows, training=False), batch_size=min(8, len(validation_rows)), shuffle=False)
    batch = next(iter(loader))
    if batch["ap"].shape[0] < 2: raise RuntimeError("pairing probe requires at least two validation pairs")
    encoder.eval(); cross_decoder.eval()
    generator = make_generator(stable_seed(SEED, FOLD, "pair_probe"))
    ap = batch["ap"].to(DEVICE); lat = batch["lat"].to(DEVICE)
    ap_mask = uniform_mask(ap.shape[0], DEVICE, generator); lat_mask = uniform_mask(lat.shape[0], DEVICE, generator)
    with torch.no_grad(), amp_context():
        ap_feature = encoder(encoder_input(ap, mean, std), ap_mask)[-1]
        lat_feature = encoder(encoder_input(lat, mean, std), lat_mask)[-1]
        target = patch_normalized_target(ap); attention = rowwise_attention_mask(DEVICE)
        paired = masked_mse(cross_decoder(ap_feature, lat_feature, attention), target, ap_mask)
        shuffled = masked_mse(cross_decoder(ap_feature, lat_feature.roll(1, 0), attention), target, ap_mask)
    return float(paired), float(shuffled)


def train_p2(train_rows, validation_rows, test_rows, manifest_meta):
    p1_export_path = ARTIFACT_ROOT / "fcmae_p1_encoder.pth"
    p1_state_path = ARTIFACT_ROOT / "fcmae_p1_best_trainstate.pth"
    if not p1_export_path.is_file() or not p1_state_path.is_file():
        raise FileNotFoundError("P2 requires this fold's approved P1 export and best trainstate")
    p1_export = torch.load(p1_export_path, map_location="cpu", weights_only=False)
    if p1_export["fold"] != FOLD or p1_export["manifest_sha256"] != manifest_meta["sha256"]:
        raise RuntimeError("P1 export fold/manifest mismatch")
    config = run_config("cross_view_P2", manifest_meta, train_rows, validation_rows, test_rows, sha256_file(p1_export_path))
    config_sha = canonical_sha256(config); write_json(ARTIFACT_ROOT / "fcmae_p2_config.json", config)

    backbone = build_backbone(pretrained=False)
    strict_load(backbone, p1_export["encoder_state"], "P1 backbone -> P2")
    mean, std = backbone.pretrained_cfg["mean"], backbone.pretrained_cfg["std"]
    encoder = FCMAEEncoder(backbone).to(DEVICE)
    fcmae_decoder = FCMAEDecoder().to(DEVICE)
    p1_trainstate = torch.load(p1_state_path, map_location="cpu", weights_only=False)
    strict_load(fcmae_decoder, p1_trainstate["modules"]["fcmae_decoder"], "P1 decoder -> P2")
    cross_decoder = CrossViewDecoder().to(DEVICE)
    modules = {"encoder": encoder, "fcmae_decoder": fcmae_decoder, "cross_decoder": cross_decoder}
    optimizer = torch.optim.AdamW([p for m in modules.values() for p in m.parameters()], lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    train_set = PairDataset(train_rows, training=True); validation_set = PairDataset(validation_rows, training=False)
    train_loader = DataLoader(train_set, batch_size=MICRO_BATCH, shuffle=True, num_workers=NUM_WORKERS)
    validation_loader = DataLoader(validation_set, batch_size=MICRO_BATCH, shuffle=False, num_workers=NUM_WORKERS)
    updates_per_epoch = math.ceil(len(train_loader) / ACCUM_STEPS)
    scheduler = scheduler_for(optimizer, P2_MAX_EPOCHS * updates_per_epoch); scaler = make_scaler()
    start_epoch, global_step, best, patience, initial_validation, history = 0, 0, float("inf"), 0, None, []

    if RESUME_P2:
        checkpoint = torch.load(RESUME_P2, map_location=DEVICE, weights_only=False)
        if checkpoint["config_sha256"] != config_sha or checkpoint["fold"] != FOLD: raise RuntimeError("P2 resume mismatch")
        for name, module in modules.items(): strict_load(module, checkpoint["modules"][name], name)
        optimizer.load_state_dict(checkpoint["optimizer"]); scheduler.load_state_dict(checkpoint["scheduler"]); scaler.load_state_dict(checkpoint["scaler"]); restore_rng_state(checkpoint["rng_state"])
        start_epoch, global_step, best = checkpoint["epoch"] + 1, checkpoint["global_step"], checkpoint["best_metric"]

    for epoch in range(start_epoch, P2_MAX_EPOCHS):
        train_set.set_epoch(epoch)
        for module in modules.values(): module.train()
        optimizer.zero_grad(set_to_none=True); totals = {"loss": 0.0, "fcmae": 0.0, "cross": 0.0, "count": 0}
        for index, batch in enumerate(train_loader):
            generator = make_generator(stable_seed(SEED, FOLD, "p2", epoch, index))
            with amp_context(): loss, pieces, _ = p2_forward(encoder, fcmae_decoder, cross_decoder, batch["ap"], batch["lat"], mean, std, generator)
            if not torch.isfinite(loss): raise FloatingPointError("non-finite P2 loss")
            scaler.scale(loss / ACCUM_STEPS).backward()
            if (index + 1) % ACCUM_STEPS == 0 or index + 1 == len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_([p for m in modules.values() for p in m.parameters()], 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step(); global_step += 1
            n = batch["ap"].shape[0]; totals["loss"] += loss.item() * n; totals["fcmae"] += float(pieces["fcmae"]) * n; totals["cross"] += float(pieces["cross"]) * n; totals["count"] += n

        for module in modules.values(): module.eval()
        validation_total, validation_count, feature_std = 0.0, 0, []
        with torch.no_grad():
            for index, batch in enumerate(validation_loader):
                generator = make_generator(stable_seed(SEED, FOLD, "p2_validation", index))
                with amp_context(): loss, _, features = p2_forward(encoder, fcmae_decoder, cross_decoder, batch["ap"], batch["lat"], mean, std, generator)
                n = batch["ap"].shape[0]; validation_total += loss.item() * n; validation_count += n; feature_std.extend([float(f.float().std()) for f in features])
        validation_loss = validation_total / validation_count
        if initial_validation is None: initial_validation = validation_loss
        record = {"epoch": epoch, "train_loss": totals["loss"] / totals["count"], "train_fcmae": totals["fcmae"] / totals["count"], "train_cross": totals["cross"] / totals["count"], "validation_loss": validation_loss, "feature_std": float(np.mean(feature_std)), "lr": optimizer.param_groups[0]["lr"]}
        history.append(record); pd.DataFrame(history).to_csv(ARTIFACT_ROOT / "fcmae_p2_history.csv", index=False)
        save_training_state(ARTIFACT_ROOT / "fcmae_p2_last_trainstate.pth", "cross_view_P2", epoch, global_step, modules, optimizer, scheduler, scaler, best, config_sha)
        if epoch % CHECKPOINT_EVERY == 0: save_training_state(ARTIFACT_ROOT / f"fcmae_p2_epoch{epoch:03d}_trainstate.pth", "cross_view_P2", epoch, global_step, modules, optimizer, scheduler, scaler, best, config_sha)
        if validation_loss < best:
            best, patience = validation_loss, 0
            save_training_state(ARTIFACT_ROOT / "fcmae_p2_best_trainstate.pth", "cross_view_P2", epoch, global_step, modules, optimizer, scheduler, scaler, best, config_sha)
        else: patience += 1
        print(record)
        if patience >= EARLY_STOP_PATIENCE: break

    best_checkpoint = torch.load(ARTIFACT_ROOT / "fcmae_p2_best_trainstate.pth", map_location="cpu", weights_only=False)
    strict_load(encoder.cpu(), best_checkpoint["modules"]["encoder"], "P2 encoder")
    strict_load(cross_decoder.cpu(), best_checkpoint["modules"]["cross_decoder"], "P2 cross decoder")
    encoder.to(DEVICE); cross_decoder.to(DEVICE)
    paired, shuffled = crossview_pairing_probe(encoder, cross_decoder, validation_rows, mean, std)
    if not best < initial_validation: raise RuntimeError("P2 validation loss did not improve")
    if not paired < shuffled: raise RuntimeError(f"pairing gate failed: paired={paired}, shuffled={shuffled}")
    export_state = {key: value.cpu() for key, value in encoder.backbone.state_dict().items()}
    export = {"schema_version": STAGE2_SCHEMA, "stage": "cross_view_P2", "fold": FOLD, "encoder_state": export_state, "config_sha256": config_sha, "manifest_sha256": manifest_meta["sha256"], "p1_encoder_sha256": sha256_file(p1_export_path)}
    torch.save(export, ARTIFACT_ROOT / "fcmae_p2_encoder.pth")
    provenance = {**config, "config_sha256": config_sha, "initial_validation_loss": initial_validation, "best_validation_loss": best, "paired_crossview_loss": paired, "shuffled_crossview_loss": shuffled, "encoder_sha256": tensor_state_sha256(export_state), "trainstate_sha256": sha256_file(ARTIFACT_ROOT / "fcmae_p2_best_trainstate.pth")}
    write_json(ARTIFACT_ROOT / "fcmae_p2_provenance.json", provenance)
    return provenance
'@

$nb01 += New-CodeCell @'
if RUN_REAL_DATA:
    ready_rows, train_rows, validation_rows, test_rows, manifest_meta = load_certified_folds(FOLD)
    print({"train": len(train_rows), "validation": len(validation_rows), "test_not_opened": len(test_rows)})
    if RUN_P1:
        print(train_p1(train_rows, validation_rows, test_rows, manifest_meta))
    if RUN_P2:
        print(train_p2(train_rows, validation_rows, test_rows, manifest_meta))
else:
    print("DATA-FREE MODE: definitions loaded. Set RUN_REAL_DATA=True only from an approved Stage 2 task brief.")
'@

$targets01 = @(
    'notebooks\modeling\01_encoder_pipeline.ipynb',
    'HPC\HPC_notebooks\modeling_HPC\01_encoder_pipeline.ipynb'
)
foreach ($target in $targets01) { Write-Notebook $target $nb01 }

