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
