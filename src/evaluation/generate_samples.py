"""
Generate fake samples from a trained DCGAN generator checkpoint.

This script loads a trained generator model and produces a specified number
of synthetic images for evaluation purposes (e.g., FID/MiFID computation).

Usage:
    python src/evaluation/generate_samples.py \
        --checkpoint outputs/checkpoint_epoch_050.pt \
        --out_dir results/fid_eval/fake_epoch_050 \
        --num_images 1000 \
        --batch_size 32
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from torchvision.utils import save_image
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.dcgan import DCGenerator256


def load_generator(checkpoint_path: str, nz: int = 100, device=None):
    """
    Load a trained generator from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint .pt file
        nz: Latent vector dimension (must match training config)
        device: torch device (cuda/cpu)
    
    Returns:
        Loaded generator model in eval mode
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize generator architecture
    G = DCGenerator256(nz=nz, ngf=64, nc=3).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'G' in checkpoint:
        G.load_state_dict(checkpoint['G'])
    elif 'generator' in checkpoint:
        G.load_state_dict(checkpoint['generator'])
    else:
        # Assume checkpoint is the state dict itself
        G.load_state_dict(checkpoint)
    
    G.eval()
    print(f"✓ Generator loaded from {checkpoint_path}")
    print(f"✓ Using device: {device}")
    
    return G, device


def generate_samples(
    generator,
    num_images: int,
    out_dir: str,
    nz: int = 100,
    batch_size: int = 32,
    device=None
):
    """
    Generate and save synthetic images using the trained generator.
    
    Args:
        generator: Trained generator model
        num_images: Total number of images to generate
        out_dir: Output directory to save images
        nz: Latent vector dimension
        batch_size: Number of images to generate per batch
        device: torch device
    """
    device = device or next(generator.parameters()).device
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating {num_images} images...")
    print(f"Output directory: {out_path}")
    
    img_idx = 0
    num_batches = (num_images + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
            # Determine batch size for this iteration
            current_batch_size = min(batch_size, num_images - img_idx)
            
            # Sample random latent vectors
            z = torch.randn(current_batch_size, nz, device=device)
            
            # Generate images
            fake_images = generator(z)
            
            # Save each image individually with consistent naming
            for i in range(current_batch_size):
                img = fake_images[i]
                # Denormalize from [-1, 1] to [0, 1]
                img = (img + 1) / 2.0
                img = torch.clamp(img, 0, 1)
                
                # Save with zero-padded filename
                img_path = out_path / f"img_{img_idx:04d}.png"
                save_image(img, img_path)
                img_idx += 1
    
    print(f"✓ Successfully generated {img_idx} images")
    print(f"✓ Saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic images from a trained DCGAN generator"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to generator checkpoint (.pt file)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for generated images"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1000,
        help="Number of images to generate (default: 1000)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for generation (default: 32)"
    )
    parser.add_argument(
        "--nz",
        type=int,
        default=100,
        help="Latent vector dimension (default: 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)"
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        print(f"✓ Random seed set to: {args.seed}")
    
    # Load generator
    generator, device = load_generator(args.checkpoint, nz=args.nz)
    
    # Generate samples
    generate_samples(
        generator=generator,
        num_images=args.num_images,
        out_dir=args.out_dir,
        nz=args.nz,
        batch_size=args.batch_size,
        device=device
    )
    
    print("\n✓ Sample generation complete!")


if __name__ == "__main__":
    main()
