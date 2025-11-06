"""
Simple Dataset Organizer (no cv2 dependency)
Organizes the actual dataset structure into the expected training structure.
"""

import os
import shutil
import json
from pathlib import Path


class SimpleDatasetOrganizer:
    def __init__(self, dataset_root: str):
        """
        Args:
            dataset_root: Path to dataset folder containing image/, cloth/, label/, etc.
        """
        self.dataset_root = Path(dataset_root)
        
        # Source folders (actual structure)
        self.src_image_dir = self.dataset_root / 'image'
        self.src_cloth_dir = self.dataset_root / 'cloth'
        self.src_label_dir = self.dataset_root / 'label'  # JSON keypoints
        self.src_agnostic_dir = self.dataset_root / 'agnostic-v3.2'  # Body masks
        self.src_openpose_dir = self.dataset_root / 'openpose-img'
        
        # Target folders (expected structure)
        self.tgt_person_dir = self.dataset_root / 'person'
        self.tgt_cloth_dir = self.dataset_root / 'cloth_organized'
        self.tgt_seg_dir = self.dataset_root / 'segmentation_masks'
        self.tgt_pose_dir = self.dataset_root / 'pose_keypoints'
        
    def create_folders(self):
        """Create target directory structure"""
        print("📁 Creating folder structure...")
        
        self.tgt_person_dir.mkdir(exist_ok=True)
        self.tgt_cloth_dir.mkdir(exist_ok=True)
        self.tgt_seg_dir.mkdir(exist_ok=True)
        self.tgt_pose_dir.mkdir(exist_ok=True)
        
        print(f"   ✓ Created: {self.tgt_person_dir}")
        print(f"   ✓ Created: {self.tgt_cloth_dir}")
        print(f"   ✓ Created: {self.tgt_seg_dir}")
        print(f"   ✓ Created: {self.tgt_pose_dir}")
        
    def copy_files(self):
        """Copy files from source to target directories"""
        print("\n📋 Copying files...")
        
        # Copy person images
        print("   → Copying person images...")
        person_count = 0
        if self.src_image_dir.exists():
            for img_file in self.src_image_dir.glob('*.jpg'):
                tgt_path = self.tgt_person_dir / img_file.name
                shutil.copy2(img_file, tgt_path)
                person_count += 1
            print(f"     ✓ Copied {person_count} person images")
        else:
            print(f"     ⚠️  Source not found: {self.src_image_dir}")
        
        # Copy cloth images
        print("   → Copying cloth images...")
        cloth_count = 0
        if self.src_cloth_dir.exists():
            for cloth_file in self.src_cloth_dir.glob('*.jpg'):
                tgt_path = self.tgt_cloth_dir / cloth_file.name
                shutil.copy2(cloth_file, tgt_path)
                cloth_count += 1
            print(f"     ✓ Copied {cloth_count} cloth images")
        else:
            print(f"     ⚠️  Source not found: {self.src_cloth_dir}")
        
        # Copy segmentation masks (from agnostic folder)
        print("   → Copying segmentation masks...")
        seg_count = 0
        if self.src_agnostic_dir.exists():
            for mask_file in self.src_agnostic_dir.glob('*.png'):
                tgt_path = self.tgt_seg_dir / mask_file.name
                shutil.copy2(mask_file, tgt_path)
                seg_count += 1
            print(f"     ✓ Copied {seg_count} segmentation masks")
        else:
            print(f"     ⚠️  Source not found: {self.src_agnostic_dir}")
        
        # Copy pose keypoints (JSON files from label folder)
        print("   → Copying pose keypoints...")
        pose_count = 0
        if self.src_label_dir.exists():
            for json_file in self.src_label_dir.glob('*.json'):
                tgt_path = self.tgt_pose_dir / json_file.name
                shutil.copy2(json_file, tgt_path)
                pose_count += 1
            print(f"     ✓ Copied {pose_count} pose keypoint files")
        else:
            print(f"     ⚠️  Source not found: {self.src_label_dir}")
        
        return person_count, cloth_count, seg_count, pose_count
        
    def create_dummy_keypoints(self, person_count: int):
        """Create dummy pose keypoint JSON files"""
        print("\n🎯 Creating pose keypoint files...")
        
        # Get all person image filenames
        person_files = sorted(self.tgt_person_dir.glob('*.jpg'))
        
        for person_file in person_files:
            # Create corresponding JSON filename
            json_name = person_file.stem + '.json'
            json_path = self.tgt_pose_dir / json_name
            
            # Create dummy keypoints (18 keypoints with x, y, confidence)
            dummy_keypoints = {
                "version": 1.0,
                "people": [
                    {
                        "pose_keypoints_2d": [0.0] * 54  # 18 keypoints * 3 (x, y, conf)
                    }
                ]
            }
            
            with open(json_path, 'w') as f:
                json.dump(dummy_keypoints, f, indent=2)
        
        print(f"   ✓ Created {len(person_files)} pose keypoint files")
        print(f"   ⚠️  Note: These are dummy keypoints (all zeros)")
        print(f"   💡 For better results, run OpenPose to generate real keypoints")
        
    def create_pairs_file(self):
        """Create pairs.txt matching person with cloth"""
        print("\n👔 Creating pairs.txt...")
        
        person_files = sorted([f.name for f in self.tgt_person_dir.glob('*.jpg')])
        cloth_files = sorted([f.name for f in self.tgt_cloth_dir.glob('*.jpg')])
        
        if len(person_files) == 0 or len(cloth_files) == 0:
            print(f"   ⚠️  Warning: person={len(person_files)}, cloth={len(cloth_files)}")
            return
        
        pairs_file = self.dataset_root / 'pairs.txt'
        
        with open(pairs_file, 'w') as f:
            # Match same names or pair sequentially
            for person_file in person_files:
                # Try to find cloth with exact same name
                if person_file in cloth_files:
                    # Use tab as delimiter to avoid issues with spaces in filenames
                    f.write(f"{person_file}\t{person_file}\n")
                else:
                    # Pair with first cloth as default
                    f.write(f"{person_file}\t{cloth_files[0]}\n")
        
        print(f"   ✓ Created pairs.txt with {len(person_files)} pairs")
        
    def verify_dataset(self):
        """Verify the organized dataset"""
        print("\n✅ Verifying dataset...")
        
        person_count = len(list(self.tgt_person_dir.glob('*.jpg')))
        cloth_count = len(list(self.tgt_cloth_dir.glob('*.jpg')))
        seg_count = len(list(self.tgt_seg_dir.glob('*.png')))
        pose_count = len(list(self.tgt_pose_dir.glob('*.json')))
        
        print(f"   📊 Person images: {person_count}")
        print(f"   📊 Cloth images: {cloth_count}")
        print(f"   📊 Segmentation masks: {seg_count}")
        print(f"   📊 Pose keypoints: {pose_count}")
        
        # Check for pairs.txt
        pairs_file = self.dataset_root / 'pairs.txt'
        if pairs_file.exists():
            with open(pairs_file, 'r') as f:
                pairs_count = len(f.readlines())
            print(f"   📊 Pairs in pairs.txt: {pairs_count}")
        
        print("\n" + "="*60)
        if person_count > 0 and cloth_count > 0:
            print("✅ Dataset organization complete!")
            print(f"   Ready for training with {person_count} samples")
        else:
            print("⚠️  Dataset organization completed with warnings")
            print("   Please check the source folders contain data")
        print("="*60)
        
    def run(self):
        """Run the complete organization process"""
        print("="*60)
        print("🚀 Virtual Try-On Dataset Organizer (Simple)")
        print("="*60)
        print(f"\n📂 Dataset root: {self.dataset_root}\n")
        
        # Step 1: Create folders
        self.create_folders()
        
        # Step 2: Copy files
        person_count, cloth_count, seg_count, pose_count = self.copy_files()
        
        # Step 3: Create dummy keypoints only if no real ones were copied
        if pose_count == 0:
            print("\n   ⚠️  No pose keypoints found, creating dummy ones...")
            self.create_dummy_keypoints(person_count)
        else:
            print(f"\n   ✓ Using {pose_count} real pose keypoint files from dataset")
        
        # Step 4: Create pairs file
        self.create_pairs_file()
        
        # Step 5: Verify
        self.verify_dataset()


if __name__ == '__main__':
    # Set dataset path
    dataset_path = 'dataset'  # Relative to script location
    
    # Create and run organizer
    organizer = SimpleDatasetOrganizer(dataset_path)
    organizer.run()
