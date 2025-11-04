# FID Evaluation Results

This directory contains evaluation results for the DCGAN model using Fréchet Inception Distance (FID).

## Directory Structure

```
fid_eval/
├── README.md              # This file
├── metrics.csv            # FID scores for all evaluated epochs
├── fid_epoch_XXX.json     # Individual JSON results per epoch
├── fake_epoch_020/        # Generated samples from epoch 20
├── fake_epoch_035/        # Generated samples from epoch 35
└── fake_epoch_050/        # Generated samples from epoch 50
```

## Metrics File Format

`metrics.csv` contains the following columns:
- **epoch**: Training epoch number
- **metric_name**: Metric type (FID)
- **score**: FID score (lower is better)
- **timestamp**: When the evaluation was performed
- **real_dir**: Path to real images used for comparison
- **fake_dir**: Path to generated images

## FID Score Interpretation

- **FID < 50**: Excellent quality - generated images are very similar to real images
- **FID 50-100**: Good quality - generated images are reasonably realistic
- **FID 100-200**: Moderate quality - noticeable differences from real images
- **FID > 200**: Poor quality - significant differences from real images