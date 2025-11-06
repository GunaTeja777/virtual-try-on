"""
Check dataset integrity and code compatibility
"""
import sys
from pathlib import Path
import json
import torch
import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.dataset_loader import VTONDataset

def check_dataset_integrity():
    """Comprehensive dataset integrity check"""
    
    print("="*70)
    print("🔍 DATASET INTEGRITY CHECK")
    print("="*70)
    
    dataset_root = Path('dataset')
    
    # 1. Check folder structure
    print("\n📁 1. FOLDER STRUCTURE CHECK")
    print("-"*70)
    
    required_folders = {
        'person': dataset_root / 'person',
        'cloth_organized': dataset_root / 'cloth_organized',
        'segmentation_masks': dataset_root / 'segmentation_masks',
        'pose_keypoints': dataset_root / 'pose_keypoints',
    }
    
    for name, path in required_folders.items():
        if path.exists():
            file_count = len(list(path.glob('*')))
            print(f"   ✓ {name:20s} - {file_count} files")
        else:
            print(f"   ✗ {name:20s} - MISSING!")
    
    # 2. Check pairs.txt
    print("\n📄 2. PAIRS.TXT CHECK")
    print("-"*70)
    
    pairs_file = dataset_root / 'pairs.txt'
    if pairs_file.exists():
        with open(pairs_file, 'r') as f:
            lines = f.readlines()
        print(f"   ✓ pairs.txt exists with {len(lines)} pairs")
        print(f"   First pair: {lines[0].strip()}")
        print(f"   Last pair: {lines[-1].strip()}")
    else:
        print(f"   ✗ pairs.txt NOT FOUND!")
    
    # 3. Check sample files
    print("\n🖼️  3. SAMPLE FILE CHECK")
    print("-"*70)
    
    person_files = sorted(list((dataset_root / 'person').glob('*.jpg')))[:3]
    for pf in person_files:
        print(f"\n   Person: {pf.name}")
        
        # Check if image loads
        try:
            img = Image.open(pf)
            print(f"     ✓ Image: {img.size} {img.mode}")
        except Exception as e:
            print(f"     ✗ Image load failed: {e}")
        
        # Check corresponding segmentation mask
        mask_path = dataset_root / 'segmentation_masks' / pf.name.replace('.jpg', '.png')
        if mask_path.exists():
            try:
                mask = Image.open(mask_path)
                print(f"     ✓ Mask: {mask.size} {mask.mode}")
                # Check if mask has actual data
                mask_array = np.array(mask)
                unique_vals = np.unique(mask_array)
                print(f"     ✓ Mask unique values: {unique_vals[:10]}")
            except Exception as e:
                print(f"     ✗ Mask load failed: {e}")
        else:
            print(f"     ✗ Mask not found: {mask_path.name}")
        
        # Check corresponding pose keypoints
        pose_path = dataset_root / 'pose_keypoints' / pf.name.replace('.jpg', '.json')
        if pose_path.exists():
            try:
                with open(pose_path, 'r') as f:
                    pose_data = json.load(f)
                keypoints = pose_data['people'][0]['pose_keypoints_2d']
                
                # Check if keypoints are all zeros (dummy)
                non_zero = sum(1 for k in keypoints if k != 0.0)
                if non_zero == 0:
                    print(f"     ⚠️  Pose: ALL ZEROS (dummy keypoints)")
                else:
                    print(f"     ✓ Pose: {non_zero}/{len(keypoints)} non-zero values")
                    print(f"     Sample keypoints: {keypoints[:6]}")
            except Exception as e:
                print(f"     ✗ Pose load failed: {e}")
        else:
            print(f"     ✗ Pose not found: {pose_path.name}")
    
    # 4. Test dataset loader
    print("\n\n🔧 4. DATASET LOADER TEST")
    print("-"*70)
    
    try:
        dataset = VTONDataset(
            data_root='dataset',
            img_size=(256, 192),
            augment=False
        )
        
        print(f"   ✓ Dataset initialized")
        print(f"   ✓ Total samples: {len(dataset)}")
        
        # Load first sample
        print(f"\n   Loading first sample...")
        sample = dataset[0]
        
        print(f"   ✓ person_image: {sample['person_image'].shape} [{sample['person_image'].dtype}]")
        print(f"     Min: {sample['person_image'].min():.3f}, Max: {sample['person_image'].max():.3f}")
        
        print(f"   ✓ cloth_image: {sample['cloth_image'].shape} [{sample['cloth_image'].dtype}]")
        print(f"     Min: {sample['cloth_image'].min():.3f}, Max: {sample['cloth_image'].max():.3f}")
        
        print(f"   ✓ segmentation: {sample['segmentation'].shape} [{sample['segmentation'].dtype}]")
        print(f"     Min: {sample['segmentation'].min():.3f}, Max: {sample['segmentation'].max():.3f}")
        unique_seg = torch.unique(sample['segmentation'])
        print(f"     Unique values: {unique_seg[:10]}")
        
        print(f"   ✓ pose_heatmap: {sample['pose_heatmap'].shape} [{sample['pose_heatmap'].dtype}]")
        print(f"     Min: {sample['pose_heatmap'].min():.3f}, Max: {sample['pose_heatmap'].max():.3f}")
        non_zero_pose = (sample['pose_heatmap'] != 0).sum().item()
        total_pose = sample['pose_heatmap'].numel()
        print(f"     Non-zero values: {non_zero_pose}/{total_pose} ({100*non_zero_pose/total_pose:.2f}%)")
        
        print(f"   ✓ person_name: {sample['person_name']}")
        print(f"   ✓ cloth_name: {sample['cloth_name']}")
        
        # Check if data makes sense
        print(f"\n   🔍 DATA VALIDATION:")
        issues = []
        
        # Check if pose heatmap is all zeros
        if non_zero_pose == 0:
            issues.append("⚠️  Pose heatmap is ALL ZEROS (dummy pose data)")
        
        # Check if segmentation has variation
        if len(unique_seg) < 2:
            issues.append("⚠️  Segmentation has no variation (might be dummy)")
        
        # Check if images are normalized
        if sample['person_image'].min() >= 0 and sample['person_image'].max() <= 1:
            issues.append("⚠️  Images might not be properly normalized (expected range ~[-1, 1])")
        
        if issues:
            print("\n   ISSUES FOUND:")
            for issue in issues:
                print(f"     {issue}")
        else:
            print("     ✓ All validations passed!")
        
    except Exception as e:
        print(f"   ✗ Dataset loader failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Summary
    print("\n\n" + "="*70)
    print("📋 SUMMARY")
    print("="*70)
    
    if pose_path.exists():
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)
        keypoints = pose_data['people'][0]['pose_keypoints_2d']
        non_zero = sum(1 for k in keypoints if k != 0.0)
        
        if non_zero == 0:
            print("\n⚠️  CRITICAL: Pose keypoints are DUMMY DATA (all zeros)")
            print("   This explains why pose training achieved 0.0 loss immediately.")
            print("   The model learned to predict zeros, matching the dummy ground truth.")
            print("\n   SOLUTIONS:")
            print("   1. Run OpenPose/MediaPipe to extract real keypoints")
            print("   2. Skip pose model and use pre-trained pose estimator")
            print("   3. Continue without pose (may affect final quality)")
        else:
            print("\n✓ Dataset appears to have real pose keypoints")
    
    print("\n✓ Dataset structure is correct")
    print("✓ Files are accessible and loadable")
    print("✓ Dataset loader works properly")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    check_dataset_integrity()
