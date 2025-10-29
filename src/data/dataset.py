"""
Dataset loader for GAN Getting Started (Monet style transfer).

This module provides PyTorch Dataset classes for loading the Monet and photo images.
Dataset source: https://www.kaggle.com/competitions/gan-getting-started/data
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Callable

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MonetPhotoDataset(Dataset):
    """
    Dataset for loading Monet paintings and photos.
    
    Args:
        data_dir: Root directory containing monet_jpg and photo_jpg folders
        domain: Either 'monet' or 'photo'
        transform: Optional transform to apply to images
        img_size: Target image size (default: 256)
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        domain: str = "monet",
        transform: Optional[Callable] = None,
        img_size: int = 256
    ):
        assert domain in ["monet", "photo"], "Domain must be 'monet' or 'photo'"
        
        self.data_dir = Path(data_dir)
        self.domain = domain
        self.img_size = img_size
        
        # Set up image directory
        self.img_dir = self.data_dir / f"{domain}_jpg"
        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"Directory {self.img_dir} not found. "
                f"Please ensure the dataset is in {self.data_dir}/"
            )
        
        # Get all image paths
        self.image_paths = sorted(list(self.img_dir.glob("*.jpg")))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.img_dir}")
        
        # Set up transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Returns:
            image: Transformed image tensor
            filename: Image filename (without extension)
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        filename = img_path.stem
        return image, filename


class PairedMonetPhotoDataset(Dataset):
    """
    Dataset that returns paired samples from both Monet and photo domains.
    Useful for unpaired image-to-image translation (CycleGAN, FastCUT, etc.)
    
    Args:
        data_dir: Root directory containing monet_jpg and photo_jpg folders
        transform: Optional transform to apply to images
        img_size: Target image size (default: 256)
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        transform: Optional[Callable] = None,
        img_size: int = 256
    ):
        self.monet_dataset = MonetPhotoDataset(data_dir, "monet", transform, img_size)
        self.photo_dataset = MonetPhotoDataset(data_dir, "photo", transform, img_size)
    
    def __len__(self) -> int:
        # Use the smaller dataset size
        return min(len(self.monet_dataset), len(self.photo_dataset))
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            monet_img: Monet painting tensor
            photo_img: Photo tensor
        """
        monet_img, _ = self.monet_dataset[idx % len(self.monet_dataset)]
        photo_img, _ = self.photo_dataset[idx % len(self.photo_dataset)]
        
        return monet_img, photo_img


def get_default_transforms(img_size: int = 256, augment: bool = False):
    """
    Get default image transforms for training or evaluation.
    
    Args:
        img_size: Target image size
        augment: Whether to apply data augmentation
    
    Returns:
        torchvision.transforms.Compose object
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


# Example usage
if __name__ == "__main__":
    # Load Monet dataset
    monet_dataset = MonetPhotoDataset(data_dir="data", domain="monet")
    print(f"Monet dataset size: {len(monet_dataset)}")
    
    # Load photo dataset
    photo_dataset = MonetPhotoDataset(data_dir="data", domain="photo")
    print(f"Photo dataset size: {len(photo_dataset)}")
    
    # Load paired dataset
    paired_dataset = PairedMonetPhotoDataset(data_dir="data")
    print(f"Paired dataset size: {len(paired_dataset)}")
    
    # Test loading a sample
    monet_img, filename = monet_dataset[0]
    print(f"Sample image shape: {monet_img.shape}, filename: {filename}")
