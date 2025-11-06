"""
Dataset loader for Virtual Try-On System
"""
import os
import json
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class VTONDataset(Dataset):
    """
    Virtual Try-On Dataset
    """
    def __init__(self, 
                 data_root: str,
                 mode: str = 'train',
                 img_size: Tuple[int, int] = (256, 192),
                 augment: bool = True):
        """
        Args:
            data_root: Root directory of dataset (should contain 'train' and 'test' folders)
            mode: 'train' or 'test'
            img_size: (height, width)
            augment: Whether to apply data augmentation
        """
        self.data_root = Path(data_root)
        self.mode = mode
        self.img_size = img_size
        self.augment = augment and mode == 'train'
        
        # Use train/test subdirectories
        mode_dir = self.data_root / mode
        
        # Define paths - new structure
        self.person_dir = mode_dir / 'image'
        self.cloth_dir = mode_dir / 'cloth'
        self.cloth_mask_dir = mode_dir / 'cloth-mask'
        self.seg_mask_dir = mode_dir / 'agnostic-v3.2'
        self.pose_img_dir = mode_dir / 'openpose_img'
        self.pose_json_dir = mode_dir / 'openpose_json'
        self.image_parse_dir = mode_dir / 'image-parse-v3'
        self.image_parse_agnostic_dir = mode_dir / 'image-parse-agnostic-'
        self.image_densepose_dir = mode_dir / 'image-densepose'
        
        # Check if directories exist
        if not self.person_dir.exists():
            print(f"⚠️  Warning: {self.person_dir} does not exist!")
        if not self.cloth_dir.exists():
            print(f"⚠️  Warning: {self.cloth_dir} does not exist!")
        
        # Load pairs
        self.pairs = self._load_pairs()
        
        # Data augmentation
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])
    
    def _load_pairs(self) -> list:
        """Load person-cloth pairs from pairs.txt"""
        # Check for pairs file in mode directory first
        mode_dir = self.data_root / self.mode
        pairs_file = mode_dir / f'{self.mode}_pairs.txt'
        
        if not pairs_file.exists():
            # Try without mode prefix
            pairs_file = mode_dir / 'pairs.txt'
        
        if not pairs_file.exists():
            # If no pairs file, match by name or create combinations
            person_files = sorted(self.person_dir.glob('*.jpg'))
            cloth_files = sorted(self.cloth_dir.glob('*.jpg'))
            
            if len(person_files) == 0 or len(cloth_files) == 0:
                print(f"⚠️  Warning: person_files={len(person_files)}, cloth_files={len(cloth_files)}")
                return []
            
            print(f"📝 No pairs file found, creating automatic pairs from {len(person_files)} images")
            pairs = []
            # Pair each person with corresponding cloth
            for person_file in person_files:
                # Try to find matching cloth
                cloth_match = self.cloth_dir / person_file.name
                if cloth_match.exists():
                    pairs.append((person_file.name, person_file.name))
                else:
                    # Use first cloth as fallback
                    pairs.append((person_file.name, cloth_files[0].name))
            
            return pairs
        
        # Load from file
        with open(pairs_file, 'r') as f:
            lines = f.readlines()
        
        pairs = []
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
                
            # Try tab delimiter first (better for filenames with spaces)
            if '\t' in line_clean:
                parts = line_clean.split('\t')
                if len(parts) >= 2:
                    pairs.append((parts[0].strip(), parts[1].strip()))
            else:
                # Fallback to space delimiter
                parts = line_clean.split()
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
                - person_image: (3, H, W)
                - cloth_image: (3, H, W)
                - segmentation: (1, H, W)
                - pose_heatmap: (num_joints, H, W)
                - person_name: str
                - cloth_name: str
        """
        person_name, cloth_name = self.pairs[idx]
        
        # Load images
        person_img = self._load_image(self.person_dir / person_name)
        cloth_img = self._load_image(self.cloth_dir / cloth_name)
        
        # Load segmentation mask
        mask_name = person_name.replace('.jpg', '.png')
        segmentation = self._load_mask(self.seg_mask_dir / mask_name)
        
        # Load pose keypoints and create heatmap - use openpose_json folder
        pose_name = person_name.replace('.jpg', '_keypoints.json')
        pose_path = self.pose_json_dir / pose_name
        if not pose_path.exists():
            # Fallback to .json
            pose_name = person_name.replace('.jpg', '.json')
            pose_path = self.pose_json_dir / pose_name
        pose_heatmap = self._load_pose(pose_path)
        
        # Apply transformations
        if self.augment:
            # Apply same transform to person and mask
            transformed = self.transform(image=person_img, mask=segmentation)
            person_tensor = transformed['image']
            mask_transformed = transformed['mask']
            
            # ToTensorV2 already returns tensor, just add channel dim and normalize
            if isinstance(mask_transformed, torch.Tensor):
                segmentation_tensor = mask_transformed.unsqueeze(0).float() / 255.0
            else:
                segmentation_tensor = torch.from_numpy(mask_transformed).unsqueeze(0).float() / 255.0
            
            # Transform cloth separately
            cloth_tensor = self.transform(image=cloth_img)['image']
        else:
            person_tensor = self.transform(image=person_img)['image']
            cloth_tensor = self.transform(image=cloth_img)['image']
            segmentation_tensor = torch.from_numpy(segmentation).unsqueeze(0).float() / 255.0
        
        # Convert pose heatmap to tensor
        pose_tensor = torch.from_numpy(pose_heatmap).float()
        
        return {
            'person_image': person_tensor,
            'cloth_image': cloth_tensor,
            'segmentation': segmentation_tensor,
            'pose_heatmap': pose_tensor,
            'person_name': person_name,
            'cloth_name': cloth_name
        }
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load and resize image"""
        if not path.exists():
            # Return dummy image if not found
            return np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        return img
    
    def _load_mask(self, path: Path) -> np.ndarray:
        """Load segmentation mask"""
        if not path.exists():
            # Try multiple naming patterns
            alternatives = [
                # Try without _mask suffix
                path.parent / path.name.replace('_mask.png', '.png'),
                # Try with .jpg extension instead of .png
                path.parent / path.name.replace('.png', '.jpg'),
                # Try without the _00 suffix (common in VITON dataset)
                path.parent / path.stem.rsplit('_', 1)[0] + '.png' if '_' in path.stem else path,
                path.parent / path.stem.rsplit('_', 1)[0] + '.jpg' if '_' in path.stem else path,
            ]
            
            for alt_path in alternatives:
                if alt_path.exists() and alt_path != path:
                    path = alt_path
                    break
            else:
                # Still not found - return dummy mask without printing (too verbose)
                return np.zeros((self.img_size[0], self.img_size[1]), dtype=np.uint8)
        
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros((self.img_size[0], self.img_size[1]), dtype=np.uint8)
        
        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]))
        return mask
    
    def _load_pose(self, path: Path) -> np.ndarray:
        """Load pose keypoints and create heatmap"""
        if not path.exists():
            # Return dummy heatmap if not found
            return np.zeros((18, self.img_size[0], self.img_size[1]), dtype=np.float32)
        
        # Load keypoints from JSON
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except:
            return np.zeros((18, self.img_size[0], self.img_size[1]), dtype=np.float32)
        
        # Parse keypoints based on format
        if 'people' in data and len(data['people']) > 0:
            keypoints = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
        elif 'keypoints' in data:
            keypoints = np.array(data['keypoints']).reshape(-1, 3)
        else:
            # Fallback
            keypoints = np.zeros((18, 3))
        
        # Create heatmaps
        heatmaps = self._create_pose_heatmap(keypoints)
        return heatmaps
    
    def _create_pose_heatmap(self, keypoints: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        """Create Gaussian heatmaps from keypoints"""
        height, width = self.img_size
        num_joints = len(keypoints)
        heatmaps = np.zeros((num_joints, height, width), dtype=np.float32)
        
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0:
                # Scale keypoints to image size
                x = int(x * width / 192) if x < 10 else int(x)
                y = int(y * height / 256) if y < 10 else int(y)
                
                if 0 <= x < width and 0 <= y < height:
                    # Create Gaussian heatmap
                    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
                    heatmap = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
                    heatmaps[i] = heatmap
        
        return heatmaps


def get_dataloader(data_root: str,
                   batch_size: int = 8,
                   mode: str = 'train',
                   img_size: Tuple[int, int] = (256, 192),
                   num_workers: int = 4,
                   shuffle: bool = True) -> DataLoader:
    """
    Create DataLoader for VTON dataset
    
    Args:
        data_root: Root directory of dataset
        batch_size: Batch size
        mode: 'train' or 'test'
        img_size: (height, width)
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader instance
    """
    dataset = VTONDataset(
        data_root=data_root,
        mode=mode,
        img_size=img_size,
        augment=(mode == 'train')
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == 'train')
    )
    
    return dataloader


if __name__ == '__main__':
    # Test dataset
    print("Testing VTON Dataset...")
    
    dataset = VTONDataset(
        data_root='dataset',
        mode='train',
        img_size=(256, 192)
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print("\nSample data:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")
        
        print("\n✅ Dataset loading successful!")
    else:
        print("⚠️  Dataset is empty. Please preprocess data first.")
