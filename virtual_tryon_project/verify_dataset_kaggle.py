"""
Verify Dataset Structure for Kaggle Training
Tests if dataset is compatible with new train/test structure
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.dataset_loader import VTONDataset
import torch

def verify_dataset(dataset_root='dataset'):
    """Verify dataset structure and compatibility"""
    
    print("="*70)
    print("🔍 DATASET VERIFICATION FOR KAGGLE")
    print("="*70)
    print()
    
    dataset_path = Path(dataset_root)
    
    # Check if dataset exists
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("\n💡 Update the path:")
        print(f"   python verify_dataset_kaggle.py --path /path/to/dataset")
        return False
    
    print(f"📁 Dataset Root: {dataset_path.absolute()}")
    print()
    
    # Check train/test structure
    train_dir = dataset_path / 'train'
    test_dir = dataset_path / 'test'
    
    print("📂 Checking folder structure...")
    
    if not train_dir.exists():
        print(f"   ❌ Missing: {train_dir}")
        return False
    else:
        print(f"   ✅ Found: train/")
    
    if test_dir.exists():
        print(f"   ✅ Found: test/")
    else:
        print(f"   ⚠️  Missing: test/ (optional)")
    
    print()
    
    # Check required subfolders in train
    required_folders = [
        'image',           # Person images
        'cloth',           # Cloth images
        'agnostic-v3.2',   # Body segmentation
        'openpose_json',   # Pose keypoints
    ]
    
    print("📂 Checking train/ subfolders...")
    all_exist = True
    for folder in required_folders:
        folder_path = train_dir / folder
        if folder_path.exists():
            count = len(list(folder_path.glob('*')))
            print(f"   ✅ {folder}/ ({count} files)")
        else:
            print(f"   ❌ {folder}/ (missing)")
            all_exist = False
    
    if not all_exist:
        print("\n❌ Required folders are missing!")
        return False
    
    print()
    
    # Test dataset loader
    print("🧪 Testing dataset loader...")
    try:
        train_dataset = VTONDataset(
            data_root=str(dataset_path),
            mode='train',
            img_size=(256, 192),
            augment=True
        )
        
        print(f"   ✅ Train dataset: {len(train_dataset)} samples")
        
        if len(train_dataset) == 0:
            print("   ❌ Dataset is empty!")
            return False
        
        # Load one sample
        print("\n📦 Loading sample...")
        sample = train_dataset[0]
        
        print("   Sample data shapes:")
        print(f"      Person: {sample['person_image'].shape}")
        print(f"      Cloth: {sample['cloth_image'].shape}")
        print(f"      Segmentation: {sample['segmentation'].shape}")
        print(f"      Pose: {sample['pose_heatmap'].shape}")
        print(f"      Names: {sample['person_name']}, {sample['cloth_name']}")
        
        # Verify data ranges
        assert sample['person_image'].min() >= -1.0, "Person image min out of range"
        assert sample['person_image'].max() <= 1.0, "Person image max out of range"
        assert sample['segmentation'].min() >= 0.0, "Segmentation min out of range"
        assert sample['segmentation'].max() <= 1.0, "Segmentation max out of range"
        
        print("\n   ✅ Data ranges valid")
        
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test test dataset if exists
    if test_dir.exists():
        print("\n🧪 Testing test dataset...")
        try:
            test_dataset = VTONDataset(
                data_root=str(dataset_path),
                mode='test',
                img_size=(256, 192),
                augment=False
            )
            print(f"   ✅ Test dataset: {len(test_dataset)} samples")
        except Exception as e:
            print(f"   ⚠️  Test dataset error: {e}")
    
    print()
    print("="*70)
    print("✅ DATASET VERIFICATION PASSED!")
    print("="*70)
    print()
    print("Your dataset is ready for Kaggle training! 🚀")
    print()
    print("Next steps:")
    print("  1. Push code to GitHub")
    print("  2. Upload this dataset to Kaggle Datasets")
    print("  3. Run training in Kaggle notebook")
    print()
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify dataset for Kaggle training')
    parser.add_argument('--path', type=str, default='dataset', help='Path to dataset root')
    
    args = parser.parse_args()
    
    success = verify_dataset(args.path)
    
    if not success:
        print("\n❌ Verification failed! Please fix the issues above.")
        sys.exit(1)
