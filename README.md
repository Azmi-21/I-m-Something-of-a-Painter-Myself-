# GAN Getting Started - COMP433

A deep learning project exploring Generative Adversarial Networks (GANs) for image generation and style transfer.

## Project Overview

This repository contains implementations and experiments with various GAN architectures, including DCGAN, FastCUT, and Latent-Diffusion models. The project focuses on training GANs on artistic datasets to generate novel images.

## Dataset

**Dataset Link:** https://www.kaggle.com/competitions/gan-getting-started/data

## Team Members

- Azmi Abidi - 40248132
- Guerlain Hitier-Lallement - 40274516
- Kaothar Reda - 40111879


## Project Structure

```
/
├── src/              # Source code for models, training, and utilities
├── notebooks/        # Jupyter notebooks for experiments and visualization
├── scripts/          # Training and evaluation scripts
├── results/          # Generated images and experiment results
├── checkpoints/      # Model checkpoints
├── logs/             # Training logs
├── data/             # Dataset directory (gitignored)
└── docs/             # Project reports and documentation
```

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset from [here](https://www.kaggle.com/competitions/gan-getting-started/data)

## Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU (recommended)

## Training

Train the DCGAN model on the Monet dataset:

```bash
python -m src.training.train_dcgan \
    --data_root data \
    --epochs 50 \
    --batch_size 4 \
    --n_critic 2
```

**Training Parameters:**
- `--data_root`: Path to dataset directory (should contain `monet_jpg/` folder)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 4, reduce if GPU memory issues)
- `--n_critic`: Train discriminator N times per generator update (default: 3)
- `--out_dir`: Output directory for samples and checkpoints (default: `outputs/`)

**Outputs:**
- Generated samples: `outputs/epoch_XXX.png` (saved every epoch)
- Model checkpoints: `outputs/checkpoint_epoch_XXX.pt` (saved every 5 epochs)

## Evaluation

### Step 1: Generate Samples from Trained Model

Generate synthetic images from a trained checkpoint for evaluation:

```bash
python src/evaluation/generate_samples.py \
    --checkpoint outputs/checkpoint_epoch_050.pt \
    --out_dir results/fid_eval/fake_epoch_050 \
    --num_images 1000 \
    --batch_size 32
```

**Parameters:**
- `--checkpoint`: Path to trained generator checkpoint
- `--out_dir`: Output directory for generated images
- `--num_images`: Number of images to generate (default: 1000)
- `--batch_size`: Batch size for generation (default: 32)
- `--nz`: Latent vector dimension (default: 100)
- `--seed`: Random seed for reproducibility (optional)

### Step 2: Compute FID Score

Evaluate image quality using Fréchet Inception Distance (FID):

```bash
python src/evaluation/evaluate_mifid.py \
    --real_dir data/monet_jpg \
    --fake_dir results/fid_eval/fake_epoch_050 \
    --batch_size 32
```

**Parameters:**
- `--real_dir`: Directory containing real images (ground truth)
- `--fake_dir`: Directory containing generated images
- `--batch_size`: Batch size for processing (default: 32)
- `--output_file`: Output CSV file (default: `results/fid_eval/metrics.csv`)
- `--checkpoint_epoch`: Epoch number (auto-extracted from path if not provided)

**FID Score Interpretation:**
- **FID < 50**: Excellent quality
- **FID 50-100**: Good quality
- **FID 100-200**: Moderate quality
- **FID > 200**: Poor quality

### Example: Evaluate Multiple Epochs

```bash
# Generate samples for epoch 20
python src/evaluation/generate_samples.py \
    --checkpoint outputs/checkpoint_epoch_020.pt \
    --out_dir results/fid_eval/fake_epoch_020 \
    --num_images 1000

# Evaluate epoch 20
python src/evaluation/evaluate_mifid.py \
    --real_dir data/monet_jpg \
    --fake_dir results/fid_eval/fake_epoch_020

# Generate samples for epoch 35
python src/evaluation/generate_samples.py \
    --checkpoint outputs/checkpoint_epoch_035.pt \
    --out_dir results/fid_eval/fake_epoch_035 \
    --num_images 1000

# Evaluate epoch 35
python src/evaluation/evaluate_mifid.py \
    --real_dir data/monet_jpg \
    --fake_dir results/fid_eval/fake_epoch_035

# Generate samples for epoch 50
python src/evaluation/generate_samples.py \
    --checkpoint outputs/checkpoint_epoch_050.pt \
    --out_dir results/fid_eval/fake_epoch_050 \
    --num_images 1000

# Evaluate epoch 50
python src/evaluation/evaluate_mifid.py \
    --real_dir data/monet_jpg \
    --fake_dir results/fid_eval/fake_epoch_050
```

**Results:**
- All FID scores are saved to `results/fid_eval/metrics.csv`
- Individual JSON results: `results/fid_eval/fid_epoch_XXX.json`
