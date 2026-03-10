import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    Resized,
    Orientationd,
)
from monai.data import Dataset, DataLoader

def main():
    # Path to the raw DICOM files for Case1
    data_dir = r"c:\Users\Chan Zheng Shao\OneDrive\Desktop\Github Repo\TestProject\TestProject\data\raw\PartLeft\Case1"
    
    print(f"Reading DICOM series from: {data_dir}")
    
    # -------------------------------------------------------------------------
    # 1. SimpleITK native reading (to demonstrate vanilla loading)
    # -------------------------------------------------------------------------
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(data_dir)
    reader.SetFileNames(dicom_names)
    
    # Load image
    sitk_image = reader.Execute()
    
    print("\n--- Original Image Properties ---")
    print(f"Size: {sitk_image.GetSize()}")
    print(f"Spacing: {sitk_image.GetSpacing()}")
    print(f"Direction: {sitk_image.GetDirection()}")
    print(f"Origin: {sitk_image.GetOrigin()}")
    
    # -------------------------------------------------------------------------
    # 2. MONAI Pipeline (The elegant, modern way)
    # -------------------------------------------------------------------------
    print("\n--- MONAI Pre-processing Pipeline ---")
    
    # Define our standardisations
    # 1. Load the image. MONAI uses ITK-based readers under the hood.
    # 2. Add a channel dimension (C, H, W, D) which models expect.
    # 3. Reorient to standard RAS (Right, Anterior, Superior).
    # 4. Spacing: Resample to isotropic 1.0mm x 1.0mm x 1.0mm voxels.
    # 5. ScaleIntensityRanged: 
    #    - Hounsfield Unit (HU) conversion is typically handled automatically by the reader for CTs.
    #    - Windowing: Let's assume soft tissue window (-160 to 240 HU standard, or abdominal -150, 250).
    #      We clip values outside this window to Min/Max, then scale to [0, 1].
    # 6. Resized: Force it to a uniform tensor shape for our network (e.g. 128x128x128).
    
    pipeline = Compose([
        LoadImaged(keys=["image"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"], 
            pixdim=(1.0, 1.0, 1.0), 
            mode=("bilinear")
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-160.0,
            a_max=240.0,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        Resized(
            keys=["image"], 
            spatial_size=(128, 128, 64), # Example target network shape
            mode="trilinear"
        )
    ])
    
    # Prepare data dictionary. MONAI pipelines often use dictionary transforms.
    # We point it to the directory containing the DICOM series.
    data = [{"image": data_dir}]
    
    # Apply the pipeline
    processed_dict = pipeline(data[0])
    
    processed_tensor = processed_dict["image"]
    
    print("\n--- Processed Image Properties ---")
    print(f"Shape: {processed_tensor.shape} (Channels, X, Y, Z)")
    print(f"Min Value Context: {processed_tensor.min():.4f}")
    print(f"Max Value Context: {processed_tensor.max():.4f}")
    
    print("\nPipeline execution successful. Tensors are ready for modelling!")

if __name__ == "__main__":
    main()
