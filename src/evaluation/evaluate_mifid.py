"""
Evaluate generated images using Fréchet Inception Distance (FID) metric.

FID measures the quality and diversity of generated images by comparing
the distribution of real and fake images in Inception feature space.
Lower FID scores indicate better image quality and diversity.

Usage:
    python src/evaluation/evaluate_mifid.py \
        --real_dir data/monet_jpg \
        --fake_dir results/fid_eval/fake_epoch_050 \
        --batch_size 32
"""

import os
import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics not found. Install with: pip install torchmetrics")


class ImageFolderDataset(Dataset):
    """Simple dataset to load images from a directory."""
    
    def __init__(self, image_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Collect all image files
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            self.image_paths.extend(list(self.image_dir.glob(ext)))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Found {len(self.image_paths)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


def get_image_transform(img_size: int = 299):
    """
    Get transform for FID computation.
    
    Inception v3 expects 299x299 images normalized to [0, 1].
    
    Args:
        img_size: Target image size (default: 299 for Inception v3)
    
    Returns:
        torchvision transform
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # Images should be in [0, 1] range for FID computation
    ])


def compute_fid(
    real_dir: str,
    fake_dir: str,
    batch_size: int = 32,
    device=None,
    img_size: int = 299
) -> float:
    """
    Compute Fréchet Inception Distance between real and fake images.
    
    Args:
        real_dir: Directory containing real images
        fake_dir: Directory containing generated/fake images
        batch_size: Batch size for processing
        device: torch device (cuda/cpu)
        img_size: Image size for Inception network (default: 299)
    
    Returns:
        FID score (lower is better)
    """
    if not TORCHMETRICS_AVAILABLE:
        raise ImportError(
            "torchmetrics is required for FID computation. "
            "Install with: pip install torchmetrics"
        )
    
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Initialize FID metric
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    # Create datasets and dataloaders
    transform = get_image_transform(img_size)
    
    real_dataset = ImageFolderDataset(real_dir, transform=transform)
    fake_dataset = ImageFolderDataset(fake_dir, transform=transform)
    
    real_loader = DataLoader(
        real_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    
    fake_loader = DataLoader(
        fake_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    
    # Process real images
    print("\nProcessing real images...")
    for batch in tqdm(real_loader, desc="Real images"):
        batch = batch.to(device)
        # Scale to [0, 255] uint8 range as expected by torchmetrics FID
        batch = (batch * 255).to(torch.uint8)
        fid.update(batch, real=True)
    
    # Process fake images
    print("\nProcessing fake images...")
    for batch in tqdm(fake_loader, desc="Fake images"):
        batch = batch.to(device)
        # Scale to [0, 255] uint8 range
        batch = (batch * 255).to(torch.uint8)
        fid.update(batch, real=False)
    
    # Compute FID score
    print("\nComputing FID score...")
    fid_score = fid.compute().item()
    
    return fid_score


def save_metrics(
    fid_score: float,
    epoch: int,
    output_file: str,
    real_dir: str,
    fake_dir: str
):
    """
    Save FID metrics to CSV file.
    
    Args:
        fid_score: Computed FID score
        epoch: Epoch number (extracted from fake_dir name)
        output_file: Path to output CSV file
        real_dir: Real images directory
        fake_dir: Fake images directory
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists to determine if we need to write header
    file_exists = output_path.exists()
    
    # Prepare row data
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        'epoch': epoch,
        'metric_name': 'FID',
        'score': f"{fid_score:.4f}",
        'timestamp': timestamp,
        'real_dir': real_dir,
        'fake_dir': fake_dir
    }
    
    # Write to CSV
    with open(output_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    print(f"\n✓ Metrics saved to: {output_path}")
    
    # Also save as JSON for easy reading
    json_path = output_path.parent / f"fid_epoch_{epoch:03d}.json"
    with open(json_path, 'w') as f:
        json.dump(row, f, indent=2)
    
    print(f"✓ JSON saved to: {json_path}")


def extract_epoch_from_path(path: str) -> int:
    """
    Extract epoch number from directory path.
    
    Examples:
        'fake_epoch_050' -> 50
        'results/fid_eval/fake_epoch_035' -> 35
    
    Args:
        path: Directory path containing epoch number
    
    Returns:
        Epoch number, or 0 if not found
    """
    import re
    match = re.search(r'epoch[_-]?(\d+)', path, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Compute FID (Fréchet Inception Distance) for generated images"
    )
    parser.add_argument(
        "--real_dir",
        type=str,
        required=True,
        help="Directory containing real images"
    )
    parser.add_argument(
        "--fake_dir",
        type=str,
        required=True,
        help="Directory containing generated/fake images"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/fid_eval/metrics.csv",
        help="Output CSV file for metrics (default: results/fid_eval/metrics.csv)"
    )
    parser.add_argument(
        "--checkpoint_epoch",
        type=int,
        default=None,
        help="Epoch number (optional, auto-extracted from fake_dir if not provided)"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=299,
        help="Image size for Inception network (default: 299)"
    )
    
    args = parser.parse_args()
    
    # Extract epoch number
    epoch = args.checkpoint_epoch
    if epoch is None:
        epoch = extract_epoch_from_path(args.fake_dir)
    
    print("=" * 70)
    print("FID Evaluation")
    print("=" * 70)
    print(f"Real images:  {args.real_dir}")
    print(f"Fake images:  {args.fake_dir}")
    print(f"Epoch:        {epoch}")
    print(f"Batch size:   {args.batch_size}")
    print("=" * 70)
    
    # Compute FID
    fid_score = compute_fid(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    
    # Print results
    print("\n" + "=" * 70)
    print(f"FID Score: {fid_score:.4f}")
    print("=" * 70)
    print("\nInterpretation:")
    print("  • Lower FID = Better image quality and diversity")
    print("  • FID < 50:  Excellent quality")
    print("  • FID 50-100: Good quality")
    print("  • FID > 100: Poor quality")
    print("=" * 70)
    
    # Save metrics
    save_metrics(
        fid_score=fid_score,
        epoch=epoch,
        output_file=args.output_file,
        real_dir=args.real_dir,
        fake_dir=args.fake_dir
    )
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
