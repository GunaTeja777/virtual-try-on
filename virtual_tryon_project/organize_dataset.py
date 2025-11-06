"""
Custom dataset organizer for your actual dataset structure
Adapts the existing dataset to work with the training pipeline
"""
import os
import shutil
from pathlib import Path
import json
import numpy as np
import cv2
from tqdm import tqdm


class DatasetOrganizer:
    """
    Organize your dataset into the expected structure for training
    
    Your current structure:
    - image/ (person images)
    - cloth/ (cloth images)
    - label/ (segmentation masks)
    - openpose-img/ (pose renderings)
    - agnostic-v3.2/ (body masks)
    - cloth-mask/ (cloth masks)
    """
    
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        
        # Source folders (your actual data)
        self.image_dir = self.dataset_root / 'image'
        self.cloth_dir = self.dataset_root / 'cloth'
        self.label_dir = self.dataset_root / 'label'
        self.openpose_dir = self.dataset_root / 'openpose-img'
        self.agnostic_dir = self.dataset_root / 'agnostic-v3.2'
        self.cloth_mask_dir = self.dataset_root / 'cloth-mask'
        
        # Target folders (for training)
        self.person_dir = self.dataset_root / 'person'
        self.segm_dir = self.dataset_root / 'segmentation_masks'
        self.pose_dir = self.dataset_root / 'pose_keypoints'
        
        print(f"📂 Dataset root: {self.dataset_root}")
    
    def create_symlinks(self):
        """
        Create symbolic links or copies to organize data without duplication
        """
        print("\n🔗 Creating organized structure...")
        
        # Create directories
        self.person_dir.mkdir(exist_ok=True)
        self.segm_dir.mkdir(exist_ok=True)
        self.pose_dir.mkdir(exist_ok=True)
        
        # Get all person image files
        person_files = sorted(self.image_dir.glob('*.jpg'))
        
        print(f"Found {len(person_files)} person images")
        
        for img_file in tqdm(person_files, desc="Organizing person images"):
            # Extract base name (e.g., "img 1" from "img 1.jpg")
            base_name = img_file.stem
            
            # Copy person image
            target_person = self.person_dir / f"{base_name}.jpg"
            if not target_person.exists():
                shutil.copy2(img_file, target_person)
            
            # Copy segmentation mask
            seg_file = self.label_dir / f"{base_name}.png"
            if seg_file.exists():
                target_seg = self.segm_dir / f"{base_name}_mask.png"
                if not target_seg.exists():
                    shutil.copy2(seg_file, target_seg)
            
            # Create pose keypoint JSON from rendered image
            pose_rendered = self.openpose_dir / f"{base_name}_rendered.png"
            if pose_rendered.exists():
                self.create_dummy_keypoints(base_name)
        
        print("✅ Organization complete!")
    
    def create_dummy_keypoints(self, base_name: str):
        """
        Create dummy keypoint JSON files
        In production, you'd extract keypoints from the rendered images
        or use OpenPose to generate them
        """
        pose_json = self.pose_dir / f"{base_name}.json"
        
        if not pose_json.exists():
            # Create dummy 18-keypoint structure (COCO format)
            # In real usage, extract from openpose-img or run OpenPose
            dummy_keypoints = {
                "people": [{
                    "pose_keypoints_2d": [0.0] * 54  # 18 keypoints * 3 (x, y, confidence)
                }]
            }
            
            with open(pose_json, 'w') as f:
                json.dump(dummy_keypoints, f)
    
    def create_pairs_file(self):
        """
        Create pairs.txt mapping person to cloth
        """
        print("\n📝 Creating pairs file...")
        
        person_files = sorted(self.person_dir.glob('*.jpg'))
        cloth_files = sorted(self.cloth_dir.glob('*.jpg'))
        
        pairs = []
        
        # Strategy: Match by index or name
        # For "img 1.jpg" person, try "img 1.jpg" cloth first, then others
        for person_file in person_files:
            person_name = person_file.name
            base_name = person_file.stem
            
            # Try to find matching cloth
            cloth_match = self.cloth_dir / person_file.name
            if cloth_match.exists():
                cloth_name = cloth_match.name
            else:
                # Use first available cloth or cycle through
                cloth_idx = len(pairs) % len(cloth_files)
                cloth_name = cloth_files[cloth_idx].name
            
            pairs.append(f"{person_name} {cloth_name}\n")
        
        # Write pairs file
        pairs_file = self.dataset_root / 'pairs.txt'
        with open(pairs_file, 'w') as f:
            f.writelines(pairs)
        
        print(f"✅ Created {len(pairs)} pairs")
    
    def verify_dataset(self):
        """
        Verify all required files are present
        """
        print("\n🔍 Verifying dataset...")
        
        person_count = len(list(self.person_dir.glob('*.jpg')))
        cloth_count = len(list(self.cloth_dir.glob('*.jpg')))
        segm_count = len(list(self.segm_dir.glob('*.png')))
        pose_count = len(list(self.pose_dir.glob('*.json')))
        
        print(f"✓ Person images: {person_count}")
        print(f"✓ Cloth images: {cloth_count}")
        print(f"✓ Segmentation masks: {segm_count}")
        print(f"✓ Pose keypoints: {pose_count}")
        
        if person_count == 0:
            print("⚠️  No person images found!")
        if cloth_count == 0:
            print("⚠️  No cloth images found!")
        
        return person_count > 0 and cloth_count > 0
    
    def run(self):
        """
        Run complete organization
        """
        print("="*60)
        print("🚀 DATASET ORGANIZER")
        print("="*60)
        
        # Check source folders exist
        if not self.image_dir.exists():
            print(f"❌ Image folder not found: {self.image_dir}")
            return False
        
        if not self.cloth_dir.exists():
            print(f"❌ Cloth folder not found: {self.cloth_dir}")
            return False
        
        # Organize data
        self.create_symlinks()
        self.create_pairs_file()
        
        # Verify
        success = self.verify_dataset()
        
        if success:
            print("\n" + "="*60)
            print("✅ DATASET READY FOR TRAINING!")
            print("="*60)
            print("\nYou can now run training:")
            print("  python main.py --dataset_path dataset --skip_preprocessing")
        
        return success


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize your dataset')
    parser.add_argument('--dataset_path', type=str, 
                       default='d:/My works/Projects/virtual try on/virtual_tryon_project/dataset',
                       help='Path to dataset root')
    
    args = parser.parse_args()
    
    organizer = DatasetOrganizer(args.dataset_path)
    organizer.run()
