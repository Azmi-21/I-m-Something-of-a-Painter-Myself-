# I'm Something of a Painter Myself - CycleGAN for Monet Style Transfer

COMP 433 - Project  
Section: N

## Authors
- Azmi Abidi - 40248132
- Guerlain Hitier Lallement - 40274516
- Kaothar Reda - 40111879

## Project Overview

This project implements a **CycleGAN** architecture using PyTorch for unpaired image-to-image translation, transforming photographs into Monet-style paintings. Developed for the [Kaggle "I'm Something of a Painter Myself"](https://www.kaggle.com/competitions/gan-getting-started) competition.

### Key Features

- **ResNet-9 Generator**: 9 residual blocks with instance normalization for high-quality 256×256 image generation
- **70×70 PatchGAN Discriminator**: Classifies overlapping patches as real/fake for better texture details
- **Cycle Consistency Loss**: Ensures content preservation during style transfer
- **Identity Loss**: Helps preserve color composition
- **FID Evaluation**: Fréchet Inception Distance scoring for quality assessment

## Requirements

### Kaggle Environment (Recommended)
- **GPU**: Tesla T4 (available free on Kaggle)
- **Runtime**: ~80 minutes for full training

### Software Dependencies
All dependencies are pre-installed on Kaggle. The notebook installs any additional packages automatically:
```
PyTorch >= 2.0
torchvision >= 0.15
torchmetrics[image] >= 1.0
torch-fidelity >= 0.3
numpy, Pillow, matplotlib, tqdm
```

## Dataset

### Obtaining the Dataset

The dataset is provided by the Kaggle competition:

1. **On Kaggle** (Recommended):
   - The dataset is automatically available when you add the competition data
   - Paths are pre-configured in the notebook:
     ```python
     MONET_DIR = '/kaggle/input/gan-getting-started/monet_jpg'
     PHOTO_DIR = '/kaggle/input/gan-getting-started/photo_jpg'
     ```

2. **Download Link**:
   - Visit: https://www.kaggle.com/competitions/gan-getting-started/data
   - Or use Kaggle CLI:
     ```bash
     kaggle competitions download -c gan-getting-started
     ```

### Dataset Contents
- **Monet paintings**: 300 images (256×256 pixels)
- **Photographs**: 7,038 images (256×256 pixels)
- **Format**: JPEG

## How to Train the Model (on Kaggle)

### Step 1: Set Up Kaggle Notebook

1. Go to [Kaggle.com](https://www.kaggle.com) and sign in
2. Navigate to the competition: [I'm Something of a Painter Myself](https://www.kaggle.com/competitions/gan-getting-started)
3. Click **"Code"** → **"New Notebook"**

### Step 2: Configure the Environment

1. In your notebook settings (right sidebar):
   - **Accelerator**: Select **GPU T4 x2**
   - **Internet**: Enable **Internet access**

### Step 3: Add the Dataset

1. Click **"Add data"** (right sidebar)
2. Select **"Competition Data"**
3. Choose **"gan-getting-started"**

### Step 4: Upload and Run the Notebook

1. Upload `notebooks/model.ipynb` to Kaggle:
   - Click **"File"** → **"Import Notebook"**
   - Upload the notebook file

2. **Run all cells** (Shift+Enter or "Run All")

### Step 5: Training Outputs

The notebook automatically generates:
- **Checkpoints**: Saved every 5 epochs to `/kaggle/working/checkpoints/`
- **Sample images**: Saved every epoch to `/kaggle/working/samples/`
- **Submission file**: `images.zip` (7,000 generated images)

### Expected Training Time
- **Kaggle Tesla T4**: ~80 minutes

## How to Validate the Model

### FID Score Evaluation

The notebook automatically computes FID (Fréchet Inception Distance) after training:

```python
# FID is computed automatically in the notebook
# Compares 300 real Monet images vs 300 generated images
fid_score = compute_fid(
    real_dir='/kaggle/input/gan-getting-started/monet_jpg',
    fake_dir='/kaggle/working/images',
    num_samples=300
)
```

### Running on Sample Test Dataset

The notebook automatically:
1. Generates 7,000 Monet-style images from the photo dataset
2. Saves them to `/kaggle/working/images/`
3. Creates `images.zip` for Kaggle submission

## Kaggle Submission

After training completes:

1. The notebook generates `images.zip` containing 7,000 Monet-style images
2. Go to the competition page → **"Submit Predictions"**
3. Upload `/kaggle/working/images.zip`
4. Your MiFID score will be calculated automatically


## References

- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) (Zhu et al., 2017)
- [Kaggle Competition: I'm Something of a Painter Myself](https://www.kaggle.com/competitions/gan-getting-started)
- [PyTorch CycleGAN Implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)