"""
Preprocessing utilities for dataset preparation
"""
import os
import json
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


class DatasetPreprocessor:
    """
    Handles extraction and organization of VTON dataset
    """
    
    def __init__(self, raw_data_path: str, output_path: str, img_size: Tuple[int, int] = (256, 192)):
        """
        Args:
            raw_data_path: Path to folder containing ZIP files
            output_path: Path to organized dataset folder
            img_size: Target image size (height, width)
        """
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.img_size = img_size
        
        # Create output directories
        self.cloth_dir = self.output_path / 'cloth'
        self.person_dir = self.output_path / 'person'
        self.segm_dir = self.output_path / 'segmentation_masks'
        self.pose_dir = self.output_path / 'pose_keypoints'
        
        for dir_path in [self.cloth_dir, self.person_dir, self.segm_dir, self.pose_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def extract_zips(self, temp_dir: str = 'temp_extracted'):
        """Extract all ZIP files to temporary directory"""
        temp_path = self.raw_data_path / temp_dir
        temp_path.mkdir(exist_ok=True)
        
        zip_files = ['images.zip', 'keypoints.zip', 'segm.zip', 'densepose.zip', 'labels.zip']
        
        print("📦 Extracting ZIP files...")
        for zip_name in zip_files:
            zip_path = self.raw_data_path / zip_name
            if zip_path.exists():
                print(f"  Extracting {zip_name}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_path / zip_name.replace('.zip', ''))
            else:
                print(f"  ⚠️  {zip_name} not found, skipping...")
        
        return temp_path
    
    def organize_images(self, extracted_path: Path):
        """Organize person and cloth images"""
        print("\n🖼️  Organizing images...")
        
        images_path = extracted_path / 'images'
        if not images_path.exists():
            print("❌ Images folder not found!")
            return
        
        # Assuming structure: images/person/ and images/cloth/
        # Adjust based on actual dataset structure
        person_images = list(images_path.rglob('*person*.jpg')) + list(images_path.rglob('*person*.png'))
        cloth_images = list(images_path.rglob('*cloth*.jpg')) + list(images_path.rglob('*cloth*.png'))
        
        # If no specific folders, split by naming convention
        if not person_images and not cloth_images:
            all_images = list(images_path.rglob('*.jpg')) + list(images_path.rglob('*.png'))
            # Try to separate based on file naming or use all as person images
            person_images = all_images
        
        # Process person images
        for idx, img_path in enumerate(tqdm(person_images, desc="Processing person images")):
            img = self.load_and_resize_image(img_path)
            if img is not None:
                output_name = f"person_{idx+1:04d}.jpg"
                cv2.imwrite(str(self.person_dir / output_name), img)
        
        # Process cloth images
        for idx, img_path in enumerate(tqdm(cloth_images, desc="Processing cloth images")):
            img = self.load_and_resize_image(img_path)
            if img is not None:
                output_name = f"cloth_{idx+1:04d}.jpg"
                cv2.imwrite(str(self.cloth_dir / output_name), img)
    
    def organize_segmentation(self, extracted_path: Path):
        """Organize segmentation masks"""
        print("\n🎭 Organizing segmentation masks...")
        
        segm_path = extracted_path / 'segm'
        if not segm_path.exists():
            print("⚠️  Segmentation folder not found, skipping...")
            return
        
        mask_files = list(segm_path.rglob('*.png')) + list(segm_path.rglob('*.jpg'))
        
        for idx, mask_path in enumerate(tqdm(mask_files, desc="Processing masks")):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]))
                output_name = f"person_{idx+1:04d}_mask.png"
                cv2.imwrite(str(self.segm_dir / output_name), mask)
    
    def organize_keypoints(self, extracted_path: Path):
        """Organize pose keypoints JSON files"""
        print("\n🎯 Organizing pose keypoints...")
        
        keypoints_path = extracted_path / 'keypoints'
        if not keypoints_path.exists():
            print("⚠️  Keypoints folder not found, skipping...")
            return
        
        json_files = list(keypoints_path.rglob('*.json'))
        
        for idx, json_path in enumerate(tqdm(json_files, desc="Processing keypoints")):
            output_name = f"person_{idx+1:04d}.json"
            shutil.copy(json_path, self.pose_dir / output_name)
    
    def create_pairs_file(self):
        """Create pairs.txt mapping person to cloth"""
        print("\n📝 Creating pairs.txt...")
        
        person_files = sorted(self.person_dir.glob('*.jpg'))
        cloth_files = sorted(self.cloth_dir.glob('*.jpg'))
        
        pairs = []
        # Simple pairing: match by index
        for person_file in person_files:
            # Randomly or sequentially pair with cloth
            if cloth_files:
                cloth_file = cloth_files[len(pairs) % len(cloth_files)]
                pairs.append(f"{person_file.name} {cloth_file.name}\n")
        
        pairs_file = self.output_path / 'pairs.txt'
        with open(pairs_file, 'w') as f:
            f.writelines(pairs)
        
        print(f"✅ Created {len(pairs)} pairs")
    
    def load_and_resize_image(self, img_path: Path) -> np.ndarray:
        """Load and resize image to target size"""
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
            return img
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None
    
    def run_preprocessing(self):
        """Run complete preprocessing pipeline"""
        print("🚀 Starting dataset preprocessing...\n")
        
        # Extract ZIPs
        temp_path = self.extract_zips()
        
        # Organize data
        self.organize_images(temp_path)
        self.organize_segmentation(temp_path)
        self.organize_keypoints(temp_path)
        
        # Create pairs
        self.create_pairs_file()
        
        # Cleanup
        print("\n🧹 Cleaning up temporary files...")
        shutil.rmtree(temp_path, ignore_errors=True)
        
        print("\n✅ Preprocessing complete!")
        print(f"📂 Output directory: {self.output_path}")


def load_keypoints(json_path: str) -> np.ndarray:
    """
    Load keypoints from JSON file
    
    Returns:
        keypoints: Array of shape (num_joints, 3) with (x, y, confidence)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Adjust based on actual JSON structure
    if 'people' in data and len(data['people']) > 0:
        keypoints = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
    elif 'keypoints' in data:
        keypoints = np.array(data['keypoints']).reshape(-1, 3)
    else:
        # Fallback: assume flat array
        keypoints = np.array(data).reshape(-1, 3)
    
    return keypoints


def create_pose_heatmap(keypoints: np.ndarray, img_size: Tuple[int, int], 
                        sigma: float = 2.0) -> np.ndarray:
    """
    Create Gaussian heatmaps from keypoints
    
    Args:
        keypoints: (num_joints, 3) array with (x, y, confidence)
        img_size: (height, width)
        sigma: Gaussian kernel sigma
    
    Returns:
        heatmaps: (num_joints, height, width)
    """
    height, width = img_size
    num_joints = len(keypoints)
    heatmaps = np.zeros((num_joints, height, width), dtype=np.float32)
    
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0:
            x, y = int(x), int(y)
            if 0 <= x < width and 0 <= y < height:
                # Create Gaussian heatmap
                xx, yy = np.meshgrid(np.arange(width), np.arange(height))
                heatmap = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
                heatmaps[i] = heatmap
    
    return heatmaps


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to [-1, 1]"""
    return (img.astype(np.float32) / 127.5) - 1.0


def denormalize_image(img: np.ndarray) -> np.ndarray:
    """Denormalize image from [-1, 1] to [0, 255]"""
    return ((img + 1.0) * 127.5).astype(np.uint8)


if __name__ == '__main__':
    # Example usage
    preprocessor = DatasetPreprocessor(
        raw_data_path='path/to/raw/data',
        output_path='dataset'
    )
    preprocessor.run_preprocessing()
