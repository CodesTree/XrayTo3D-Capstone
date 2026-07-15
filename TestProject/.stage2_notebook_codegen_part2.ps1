Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function New-MarkdownCell([string]$Source) {
    [ordered]@{ cell_type = 'markdown'; metadata = [ordered]@{}; source = @($Source -split "(?<=`n)") }
}
function New-CodeCell([string]$Source, [string[]]$Tags = @()) {
    $metadata = [ordered]@{}
    if ($Tags.Count -gt 0) { $metadata.tags = $Tags }
    [ordered]@{ cell_type = 'code'; execution_count = $null; metadata = $metadata; outputs = @(); source = @($Source -split "(?<=`n)") }
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

$nb02 = @()
$nb02 += New-MarkdownCell @'
# Stage 2.2 — Fold-specific neutral shared-front-end training

This notebook loads the matching fold's approved P2 encoder, trains the complete encoder + hybrid bi-planar fusion + orthogonal 3D lift through a disposable neutral head, then exports a frozen head-free front end.

Only certified training and validation samples are opened. Training uses deterministic photometric augmentation; validation and frozen-feature hashing use clean DRRs. The final checkpoint contains no neutral-head parameters.
'@

$nb02 += New-CodeCell @'
import contextlib
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.utils.data import DataLoader, Dataset, get_worker_info
import timm

PROTOCOL_VERSION = "baseline_protocol_v1"
STAGE2_SCHEMA = "foundation_stage2_v1"
SEED = 42
FOLD = 0
RUN_REAL_DATA = False
RUN_FRONTEND_TRAINING = False
RESUME_FROM = None

PRETRAIN_MODEL = "convnextv2_tiny.fcmae"
IMAGE_SIZE = 256
TARGET_SIZE = 256
BONES = ["femur", "tibia", "patella", "fibula"]
OUTPUT_CHANNELS = [64, 128, 256, 512]
FUSION_TYPES = ["local", "local", "attention", "attention"]

EPOCHS = 40
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
CHECKPOINT_EVERY = 5
NUM_WORKERS = 4 if os.name != "nt" else 0
USE_AMP = True
USE_ACTIVATION_CHECKPOINTING = True

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
assert TARGET_SIZE == 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print({"fold": FOLD, "device": str(DEVICE), "run_real_data": RUN_REAL_DATA})
'@ @('environment-configuration')

$nb02 += New-CodeCell @'
def find_project_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "configs" / "baseline_protocol_v1.json").exists() and (candidate / "reports" / "manifests").exists():
            return candidate
    raise FileNotFoundError("project root not found")


ROOT = find_project_root(Path.cwd())
MANIFEST_PATH = ROOT / "reports" / "manifests" / "quantitative_manifest_v1.csv"
MANIFEST_META_PATH = ROOT / "reports" / "manifests" / "quantitative_manifest_v1.metadata.json"
BASELINE_CONFIG_PATH = ROOT / "configs" / "baseline_protocol_v1.json"
DATA_CONFIG_PATH = ROOT / "configs" / "data_contract_v1.json"
FOLD_ROOT = ROOT / "models" / STAGE2_SCHEMA / f"fold_{FOLD}"
P2_ENCODER_PATH = FOLD_ROOT / "fcmae_p2_encoder.pth"
P1_CONFIG_PATH = FOLD_ROOT / "fcmae_p1_config.json"


def sha256_file(path: Path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def state_sha256(state):
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode()); h.update(tensor_sha256(state[key]).encode())
    return h.hexdigest()


def stable_seed(*parts):
    token = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32)


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def augment_drr(array, sample_id, view, epoch, worker_id):
    rng = np.random.default_rng(stable_seed(SEED, FOLD, epoch, worker_id, sample_id, view))
    gamma = rng.uniform(*AUGMENTATION["gamma"])
    brightness = rng.uniform(*AUGMENTATION["brightness"])
    sigma = rng.uniform(*AUGMENTATION["gaussian_noise_sigma"])
    output = np.power(np.clip(array, 0, 1), gamma, dtype=np.float32) + np.float32(brightness)
    if sigma > 0: output += rng.normal(0, sigma, output.shape).astype(np.float32)
    return np.clip(output, 0, 1).astype(np.float32)


def load_certified_folds(fold):
    meta = json.loads(MANIFEST_META_PATH.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_CONFIG_PATH.read_text(encoding="utf-8"))
    data_contract = json.loads(DATA_CONFIG_PATH.read_text(encoding="utf-8"))
    if baseline["protocol_version"] != PROTOCOL_VERSION: raise RuntimeError("protocol mismatch")
    if not meta.get("certification_approved", False): raise RuntimeError("Stage 1 is not certified")
    if int(meta.get("ready_rows", -1)) != 71 or int(meta.get("pending_recertification_rows", -1)) != 0: raise RuntimeError("Stage 1 must have 71 ready and zero pending rows")
    if sha256_file(MANIFEST_PATH) != meta.get("sha256"): raise RuntimeError("manifest hash mismatch")
    rows = pd.read_csv(MANIFEST_PATH, dtype={"test_fold": "Int64"})
    rows = rows[rows.status.eq("ready")].copy()
    if len(rows) != 71 or rows.groupby("dataset").size().to_dict() != {"Ruikar": 13, "VSD": 58}: raise RuntimeError("certified cohort mismatch")
    rows["test_fold"] = rows.test_fold.astype(int)
    if set(rows.test_fold) != set(range(5)) or rows.groupby("subject_id").test_fold.nunique().max() != 1: raise RuntimeError("fold isolation failure")
    if rows.target_version.ne(data_contract["target_version"]).any() or rows.drr_version.ne(data_contract["drr_version"]).any(): raise RuntimeError("data version mismatch")
    rows["split"] = "train"
    rows.loc[rows.test_fold.eq(fold), "split"] = "test"
    rows.loc[rows.test_fold.eq((fold + 1) % 5), "split"] = "validation"
    by_split = {name: set(group.subject_id) for name, group in rows.groupby("split")}
    if by_split["train"] & by_split["validation"] or by_split["train"] & by_split["test"] or by_split["validation"] & by_split["test"]: raise RuntimeError("subject overlap between roles")
    for split in ("train", "validation"):
        for row in rows[rows.split.eq(split)].itertuples(index=False):
            for field in ("ap_drr_path", "lat_drr_path"):
                if not (ROOT / getattr(row, field)).is_file(): raise FileNotFoundError(f"missing certified {split} DRR: {getattr(row, field)}")
            target_dir = ROOT / row.target_path
            for bone in BONES:
                if not (target_dir / f"{row.sample_id}_{bone}.nii.gz").is_file(): raise FileNotFoundError(f"missing certified {split} target: {row.sample_id}/{bone}")
    return rows[rows.split.eq("train")].copy(), rows[rows.split.eq("validation")].copy(), rows[rows.split.eq("test")].copy(), meta


def read_drr(path):
    array = np.load(path).astype(np.float32)
    if array.shape != (256, 256) or not np.isfinite(array).all() or array.min() < -1e-6 or array.max() > 1 + 1e-6: raise ValueError(f"invalid DRR: {path}")
    return np.clip(array, 0, 1)


def load_target(row):
    target_dir = ROOT / row.target_path
    arrays, geometry = [], None
    for bone in BONES:
        image = nib.load(str(target_dir / f"{row.sample_id}_{bone}.nii.gz"))
        if image.shape != (256, 256, 256): raise ValueError(f"target shape mismatch: {row.sample_id}/{bone}")
        if tuple(nib.aff2axcodes(image.affine)) != ("L", "P", "S"): raise ValueError(f"target orientation mismatch: {row.sample_id}/{bone}")
        current_geometry = (tuple(np.round(image.affine.ravel(), 7)), tuple(np.round(image.header.get_zooms()[:3], 7)))
        if geometry is None: geometry = current_geometry
        if current_geometry != geometry: raise ValueError(f"per-bone geometry mismatch: {row.sample_id}")
        array = np.asarray(image.dataobj, dtype=np.float32)
        unique = np.unique(array)
        if not set(unique.tolist()).issubset({0.0, 1.0}) or array.sum() == 0: raise ValueError(f"non-binary or empty target: {row.sample_id}/{bone}")
        arrays.append(array)
    return torch.from_numpy(np.stack(arrays, axis=0).astype(np.float32))


class FrontEndDataset(Dataset):
    def __init__(self, rows, training):
        self.rows = rows.reset_index(drop=True); self.training = training; self.epoch = 0
    def set_epoch(self, epoch): self.epoch = int(epoch)
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows.iloc[index]
        worker = get_worker_info(); worker_id = 0 if worker is None else worker.id
        ap = read_drr(ROOT / row.ap_drr_path); lat = read_drr(ROOT / row.lat_drr_path)
        if self.training:
            ap = augment_drr(ap, row.sample_id, "ap", self.epoch, worker_id)
            lat = augment_drr(lat, row.sample_id, "lat", self.epoch, worker_id)
        return {"ap": torch.from_numpy(ap).unsqueeze(0), "lat": torch.from_numpy(lat).unsqueeze(0), "target": load_target(row), "sample_id": row.sample_id, "subject_id": row.subject_id}


seed_everything()
print("project root:", ROOT)
'@

$nb02 += New-CodeCell @'
class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.query = nn.Linear(dim, dim); self.key = nn.Linear(dim, dim); self.value = nn.Linear(dim, dim); self.scale = dim ** -0.5
    def forward(self, query_map, context_map):
        batch, channels, height, width = query_map.shape
        query = query_map.flatten(2).transpose(1, 2); context = context_map.flatten(2).transpose(1, 2)
        attention = torch.softmax(self.query(query) @ self.key(context).transpose(-2, -1) * self.scale, dim=-1)
        return (attention @ self.value(context) + query).transpose(1, 2).reshape(batch, channels, height, width)


class LocalFusion(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.mix = nn.Conv2d(2 * dim, dim, 3, padding=1)
    def forward(self, query_map, context_map): return self.mix(torch.cat([query_map, context_map], dim=1)) + query_map


class BiPlanarFrontEnd(nn.Module):
    """Hybrid bidirectional fusion followed by geometry-locked orthogonal lifting."""
    def __init__(self, encoder_state, pretrained_configuration):
        super().__init__()
        self.encoder = timm.create_model(PRETRAIN_MODEL, pretrained=False, features_only=True)
        incompatible = self.encoder.load_state_dict(encoder_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys: raise RuntimeError(f"P2 encoder strict-load failure: {incompatible}")
        self.pretrained_configuration = pretrained_configuration
        feature_channels = self.encoder.feature_info.channels()
        self.fusion = nn.ModuleList([CrossAttention(c) if kind == "attention" else LocalFusion(c) for c, kind in zip(feature_channels, FUSION_TYPES)])
        self.project_2d = nn.ModuleList([nn.Conv2d(source, target, 1) for source, target in zip(feature_channels, OUTPUT_CHANNELS)])
        self.fuse_3d = nn.ModuleList([nn.Conv3d(2 * channels, channels, 3, padding=1) for channels in OUTPUT_CHANNELS])

    @staticmethod
    def _orthogonal_lift(ap_feature, lat_feature, projection, fusion3d):
        ap = projection(ap_feature); lat = projection(lat_feature).flip(3)
        batch, channels, size, _ = ap.shape
        ap_cube = ap.permute(0, 1, 3, 2).unsqueeze(3).expand(batch, channels, size, size, size)
        lat_cube = lat.permute(0, 1, 3, 2).unsqueeze(2).expand(batch, channels, size, size, size)
        return fusion3d(torch.cat([ap_cube, lat_cube], dim=1))

    def normalize(self, raw):
        image = raw.repeat(1, 3, 1, 1)
        mean = torch.as_tensor(self.pretrained_configuration["mean"], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        std = torch.as_tensor(self.pretrained_configuration["std"], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        return (image - mean) / std

    def forward(self, ap_raw, lat_raw):
        ap_levels = self.encoder(self.normalize(ap_raw)); lat_levels = self.encoder(self.normalize(lat_raw)); lifted = []
        for ap, lat, fusion, project, fuse3d in zip(ap_levels, lat_levels, self.fusion, self.project_2d, self.fuse_3d):
            ap_refined = fusion(ap, lat); lat_refined = fusion(lat, ap)
            lifted.append(self._orthogonal_lift(ap_refined, lat_refined, project, fuse3d))
        return lifted


class NeutralBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__(); self.block = nn.Sequential(nn.Conv3d(input_channels, output_channels, 3, padding=1), nn.GroupNorm(8, output_channels), nn.ReLU(inplace=True))
    def forward(self, x): return self.block(x)


class NeutralHead(nn.Module):
    def __init__(self):
        super().__init__(); c0, c1, c2, c3 = OUTPUT_CHANNELS
        self.up3 = nn.ConvTranspose3d(c3, c2, 2, 2); self.dec3 = NeutralBlock(c2 + c2, c2)
        self.up2 = nn.ConvTranspose3d(c2, c1, 2, 2); self.dec2 = NeutralBlock(c1 + c1, c1)
        self.up1 = nn.ConvTranspose3d(c1, c0, 2, 2); self.dec1 = NeutralBlock(c0 + c0, c0)
        self.refine128 = NeutralBlock(c0, 32); self.refine192 = NeutralBlock(32, 16); self.refine256 = NeutralBlock(16, 8); self.output = nn.Conv3d(8, len(BONES), 1)
    def _run(self, block, x):
        if USE_ACTIVATION_CHECKPOINTING and self.training and x.requires_grad: return checkpoint.checkpoint(block, x, use_reentrant=False)
        return block(x)
    def forward(self, features):
        l0, l1, l2, l3 = features
        x = self._run(self.dec3, torch.cat([self.up3(l3), l2], dim=1)); x = self._run(self.dec2, torch.cat([self.up2(x), l1], dim=1)); x = self._run(self.dec1, torch.cat([self.up1(x), l0], dim=1))
        x = self._run(self.refine128, F.interpolate(x, size=(128, 128, 128), mode="trilinear", align_corners=False))
        x = self._run(self.refine192, F.interpolate(x, size=(192, 192, 192), mode="trilinear", align_corners=False))
        x = self._run(self.refine256, F.interpolate(x, size=(256, 256, 256), mode="trilinear", align_corners=False))
        return self.output(x)


class FrontEndTrainingModel(nn.Module):
    def __init__(self, front_end, neutral_head): super().__init__(); self.front_end = front_end; self.neutral_head = neutral_head
    def forward(self, ap, lat): return self.neutral_head(self.front_end(ap, lat))


def dice_bce_loss(logits, target):
    logits = logits.float(); target = target.float(); bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits).flatten(2); flattened_target = target.flatten(2); intersection = (probability * flattened_target).sum(-1)
    dice = (2 * intersection + 1.0) / (probability.sum(-1) + flattened_target.sum(-1) + 1.0)
    return 0.5 * bce + 0.5 * (1.0 - dice.mean())


@torch.no_grad()
def hard_per_bone_dice(logits, target):
    prediction = (torch.sigmoid(logits.float()) > 0.5).float().flatten(2); target = (target > 0.5).float().flatten(2); intersection = (prediction * target).sum(-1)
    return (2 * intersection + 1e-6) / (prediction.sum(-1) + target.sum(-1) + 1e-6)
'@

$nb02 += New-CodeCell @'
def amp_context():
    if not (USE_AMP and DEVICE.type == "cuda"): return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)


def make_scaler(): return torch.amp.GradScaler("cuda", enabled=USE_AMP and DEVICE.type == "cuda" and not torch.cuda.is_bf16_supported())


def capture_rng_state(): return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng_state(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["cuda"])


def build_front_end():
    if not P2_ENCODER_PATH.is_file() or not P1_CONFIG_PATH.is_file(): raise FileNotFoundError("matching fold P2 encoder/provenance is required")
    export = torch.load(P2_ENCODER_PATH, map_location="cpu", weights_only=False)
    if export.get("schema_version") != STAGE2_SCHEMA or export.get("fold") != FOLD or export.get("stage") != "cross_view_P2": raise RuntimeError("P2 export schema/fold/stage mismatch")
    p1_config = json.loads(P1_CONFIG_PATH.read_text(encoding="utf-8")); pretrained_configuration = p1_config.get("pretrained_configuration")
    if not pretrained_configuration: raise RuntimeError("P1 config does not record pretrained_configuration")
    return BiPlanarFrontEnd(export["encoder_state"], pretrained_configuration), export


def run_epoch(model, loader, optimizer, scaler, training):
    model.train(training); total_loss, count, dice_rows = 0.0, 0, []
    for batch in loader:
        ap = batch["ap"].to(DEVICE, non_blocking=True); lat = batch["lat"].to(DEVICE, non_blocking=True); target = batch["target"].to(DEVICE, non_blocking=True)
        with torch.set_grad_enabled(training):
            with amp_context(): logits = model(ap, lat); loss = dice_bce_loss(logits, target)
        if not torch.isfinite(loss): raise FloatingPointError("non-finite neutral-front-end loss")
        if training:
            optimizer.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
        total_loss += loss.item() * ap.shape[0]; count += ap.shape[0]; dice_rows.append(hard_per_bone_dice(logits, target).cpu())
    return total_loss / count, torch.cat(dice_rows, dim=0).mean(dim=0).numpy()


def frontend_config(manifest_meta, train_rows, validation_rows, test_rows, p2_sha):
    return {"schema_version": STAGE2_SCHEMA, "protocol_version": PROTOCOL_VERSION, "stage": "neutral_shared_frontend", "fold": FOLD, "seed": SEED, "manifest_sha256": manifest_meta["sha256"], "p2_encoder_sha256": p2_sha, "train_sample_ids": sorted(train_rows.sample_id.tolist()), "validation_sample_ids": sorted(validation_rows.sample_id.tolist()), "test_sample_ids_not_opened": sorted(test_rows.sample_id.tolist()), "augmentation": AUGMENTATION, "target": "four_channel_binary_occupancy", "bones": BONES, "architecture": {"fusion_types": FUSION_TYPES, "lift": "bidirectional_hybrid_orthogonal_lps", "feature_channels": OUTPUT_CHANNELS, "neutral_block": "single_conv_groupnorm8_relu", "neutral_head_discarded": True}, "hyperparameters": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer": "Adam", "learning_rate": LEARNING_RATE, "loss": "0.5_bce+0.5_soft_dice", "amp": USE_AMP, "activation_checkpointing": USE_ACTIVATION_CHECKPOINTING}, "software": {"python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__, "nibabel": nib.__version__}, "device": str(DEVICE)}


def save_trainstate(path, epoch, model, optimizer, scheduler, scaler, best_dice, best_loss, config_sha):
    torch.save({"schema_version": STAGE2_SCHEMA, "stage": "neutral_shared_frontend", "fold": FOLD, "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng_state": capture_rng_state(), "best_macro_dice": best_dice, "best_validation_loss": best_loss, "config_sha256": config_sha}, path)


def strict_load(module, state, label):
    incompatible = module.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys: raise RuntimeError(f"{label} strict-load failed: {incompatible}")


def cache_fold0_features(frozen_front_end, train_rows, config_sha, manifest_sha, frontend_sha):
    requested = ["VSD_016_Left", "Case14_PartRight"]; selected = train_rows[train_rows.sample_id.isin(requested)].copy()
    if sorted(selected.sample_id.tolist()) != sorted(requested): raise RuntimeError("fold-0 feature-cache cases must both be training rows")
    clean_set = FrontEndDataset(selected, training=False); records, manifest_records = [], []; frozen_front_end.eval()
    for index in range(len(clean_set)):
        item = clean_set[index]; ap = item["ap"].unsqueeze(0).to(DEVICE); lat = item["lat"].unsqueeze(0).to(DEVICE)
        with torch.no_grad(): first = [x.float().cpu().contiguous() for x in frozen_front_end(ap, lat)]; second = [x.float().cpu().contiguous() for x in frozen_front_end(ap, lat)]
        first_hashes = [tensor_sha256(x) for x in first]; second_hashes = [tensor_sha256(x) for x in second]
        if first_hashes != second_hashes: raise RuntimeError(f"non-deterministic frozen features for {item['sample_id']}")
        target = item["target"].to(torch.uint8).contiguous(); target_sha = tensor_sha256(target)
        records.append({"sample_id": item["sample_id"], "features": first, "target": target, "feature_sha256": first_hashes, "target_sha256": target_sha})
        manifest_records.append({"sample_id": item["sample_id"], "feature_shapes": [list(x.shape) for x in first], "feature_dtype": "float32", "feature_sha256": first_hashes, "target_shape": list(target.shape), "target_dtype": "uint8", "target_sha256": target_sha})
    cache_payload = {"schema_version": STAGE2_SCHEMA, "fold": FOLD, "manifest_sha256": manifest_sha, "frontend_checkpoint_sha256": frontend_sha, "config_sha256": config_sha, "records": records}
    cache_path = FOLD_ROOT / "frozen_feature_cache.pt"; torch.save(cache_payload, cache_path)
    cache_manifest = {"schema_version": STAGE2_SCHEMA, "fold": FOLD, "manifest_sha256": manifest_sha, "frontend_checkpoint_sha256": frontend_sha, "cache_sha256": sha256_file(cache_path), "records": manifest_records}
    (FOLD_ROOT / "frozen_feature_cache_manifest.json").write_text(json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache_manifest


def train_frontend(train_rows, validation_rows, test_rows, manifest_meta):
    FOLD_ROOT.mkdir(parents=True, exist_ok=True); front_end, _ = build_front_end(); p2_sha = sha256_file(P2_ENCODER_PATH)
    config = frontend_config(manifest_meta, train_rows, validation_rows, test_rows, p2_sha); config_sha = canonical_sha256(config)
    (FOLD_ROOT / "shared_frontend_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model = FrontEndTrainingModel(front_end, NeutralHead()).to(DEVICE)
    if not all(parameter.requires_grad for parameter in model.front_end.parameters()): raise RuntimeError("encoder/fusion/lift must all be trainable during neutral-head training")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS); scaler = make_scaler()
    train_set = FrontEndDataset(train_rows, training=True); validation_set = FrontEndDataset(validation_rows, training=False)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS); validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    start_epoch, best_dice, best_loss, history = 0, -1.0, float("inf"), []
    if RESUME_FROM:
        saved = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=False)
        if saved["fold"] != FOLD or saved["config_sha256"] != config_sha: raise RuntimeError("front-end resume fold/config mismatch")
        strict_load(model, saved["model"], "front-end training model"); optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"]); scaler.load_state_dict(saved["scaler"]); restore_rng_state(saved["rng_state"]); start_epoch, best_dice, best_loss = saved["epoch"] + 1, saved["best_macro_dice"], saved["best_validation_loss"]
    started = time.time()
    for epoch in range(start_epoch, EPOCHS):
        train_set.set_epoch(epoch); train_loss, train_dice = run_epoch(model, train_loader, optimizer, scaler, True); validation_loss, validation_dice = run_epoch(model, validation_loader, optimizer, scaler, False); scheduler.step(); macro = float(validation_dice.mean())
        record = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss, "train_macro_dice": float(train_dice.mean()), "validation_macro_dice": macro, **{f"validation_dice_{bone}": float(validation_dice[i]) for i, bone in enumerate(BONES)}, "lr": optimizer.param_groups[0]["lr"], "elapsed_seconds": round(time.time() - started, 1)}
        history.append(record); pd.DataFrame(history).to_csv(FOLD_ROOT / "shared_frontend_history.csv", index=False)
        save_trainstate(FOLD_ROOT / "shared_frontend_last_trainstate.pth", epoch, model, optimizer, scheduler, scaler, best_dice, best_loss, config_sha)
        if epoch % CHECKPOINT_EVERY == 0: save_trainstate(FOLD_ROOT / f"shared_frontend_epoch{epoch:03d}_trainstate.pth", epoch, model, optimizer, scheduler, scaler, best_dice, best_loss, config_sha)
        if macro > best_dice or (math.isclose(macro, best_dice) and validation_loss < best_loss):
            best_dice, best_loss = macro, validation_loss; save_trainstate(FOLD_ROOT / "shared_frontend_best_trainstate.pth", epoch, model, optimizer, scheduler, scaler, best_dice, best_loss, config_sha)
        print(record)
    best = torch.load(FOLD_ROOT / "shared_frontend_best_trainstate.pth", map_location="cpu", weights_only=False); strict_load(model.cpu(), best["model"], "best neutral model")
    front_end_state = {key: value.cpu() for key, value in model.front_end.state_dict().items()}
    if any("neutral_head" in key for key in front_end_state): raise RuntimeError("neutral-head key leaked into export")
    export = {"schema_version": STAGE2_SCHEMA, "stage": "shared_frontend", "fold": FOLD, "front_end_state": front_end_state, "config_sha256": config_sha, "manifest_sha256": manifest_meta["sha256"], "p2_encoder_sha256": p2_sha}
    export_path = FOLD_ROOT / "shared_frontend.pth"; torch.save(export, export_path)
    reloaded, _ = build_front_end(); strict_load(reloaded, export["front_end_state"], "head-free front end")
    for parameter in reloaded.parameters(): parameter.requires_grad = False
    if any(parameter.requires_grad for parameter in reloaded.parameters()): raise RuntimeError("front end did not freeze")
    reloaded = reloaded.to(DEVICE).eval(); frontend_sha = sha256_file(export_path); cache_manifest = cache_fold0_features(reloaded, train_rows, config_sha, manifest_meta["sha256"], frontend_sha) if FOLD == 0 else None
    provenance = {**config, "config_sha256": config_sha, "best_validation_macro_dice": best_dice, "best_validation_loss": best_loss, "front_end_state_sha256": state_sha256(front_end_state), "checkpoint_sha256": frontend_sha, "trainstate_sha256": sha256_file(FOLD_ROOT / "shared_frontend_best_trainstate.pth"), "feature_cache": cache_manifest}
    (FOLD_ROOT / "shared_frontend_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return provenance
'@

$nb02 += New-CodeCell @'
if RUN_REAL_DATA:
    train_rows, validation_rows, test_rows, manifest_meta = load_certified_folds(FOLD)
    print({"train": len(train_rows), "validation": len(validation_rows), "test_not_opened": len(test_rows)})
    if RUN_FRONTEND_TRAINING: print(train_frontend(train_rows, validation_rows, test_rows, manifest_meta))
else:
    print("DATA-FREE MODE: front-end definitions loaded. Real training remains blocked on Stage 1 PASS.")
'@

$targets02 = @('notebooks\modeling\02_frontend_pretrain.ipynb', 'HPC\HPC_notebooks\modeling_HPC\02_frontend_pretrain.ipynb')
foreach ($target in $targets02) { Write-Notebook $target $nb02 }
