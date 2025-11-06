"""
Test script to verify dataset loading works correctly
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from utils.dataset_loader import VTONDataset
    
    print("="*60)
    print("🧪 Testing Dataset Loader")
    print("="*60)
    
    # Create dataset
    print("\n📂 Loading dataset...")
    dataset = VTONDataset(
        data_root='dataset',
        img_size=(256, 192)
    )
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   📊 Total samples: {len(dataset)}")
    
    if len(dataset) > 0:
        print(f"\n🔍 Testing first sample...")
        sample = dataset[0]
        
        print(f"   ✓ Person image: {sample['person_image'].shape}")
        print(f"   ✓ Cloth image: {sample['cloth_image'].shape}")
        print(f"   ✓ Segmentation mask: {sample['segmentation'].shape}")
        print(f"   ✓ Pose heatmap: {sample['pose_heatmap'].shape}")
        print(f"   ✓ Person name: {sample['person_name']}")
        print(f"   ✓ Cloth name: {sample['cloth_name']}")
        
        print(f"\n✅ All checks passed!")
        print("="*60)
        print("🎉 Dataset is ready for training!")
        print("="*60)
    else:
        print("\n⚠️  Warning: Dataset has 0 samples")
        print("   Check that the dataset folders contain files")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
