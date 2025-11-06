"""
Train All Models - Kaggle Optimized Version
Trains all 5 models sequentially with proper dataset structure
"""
import os
import sys
from pathlib import Path

# Add project root to path FIRST (before importing numpy-dependent packages)
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# NumPy check (after path setup)
print("🔧 Checking environment...")
try:
    import numpy as np
    print(f"✅ NumPy {np.__version__}")
    
    # Warning if NumPy 2.x, but don't exit - user already fixed it in notebook
    if np.__version__.startswith('2.'):
        print("⚠️  Warning: NumPy 2.x detected")
        print("   If training fails, restart kernel and fix NumPy in notebook first.")
except ImportError:
    print("⚠️  NumPy not found")

import torch

print("="*80)
print("🚀 VIRTUAL TRY-ON - KAGGLE TRAINING PIPELINE")
print("="*80)
print()

# Check GPU availability
if torch.cuda.is_available():
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  No GPU detected! Training will be slow.")
print()

# Configuration
DATASET_ROOT = '/kaggle/input/your-dataset-name'  # Update this path in Kaggle
CHECKPOINT_DIR = 'checkpoints'
BATCH_SIZE = 16  # Kaggle has good GPUs, can use larger batch
NUM_WORKERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training epochs
EPOCHS_CONFIG = {
    'segmentation': 50,
    'pose': 50,
    'warp': 30,
    'tryon': 30,
    'refine': 20
}

print(f"📁 Dataset Root: {DATASET_ROOT}")
print(f"💾 Checkpoint Dir: {CHECKPOINT_DIR}")
print(f"🎮 Device: {DEVICE}")
print(f"📦 Batch Size: {BATCH_SIZE}")
print()

# Check if running in Kaggle
IS_KAGGLE = os.path.exists('/kaggle')
if IS_KAGGLE:
    print("✅ Running in Kaggle environment")
    # In Kaggle, update dataset path
    if os.path.exists('/kaggle/input'):
        dataset_folders = os.listdir('/kaggle/input')
        if len(dataset_folders) > 0:
            DATASET_ROOT = f'/kaggle/input/{dataset_folders[0]}'
            print(f"   Auto-detected dataset: {DATASET_ROOT}")
else:
    print("💻 Running locally")
    DATASET_ROOT = 'dataset'  # Use local dataset

print()

def train_model(model_name, train_script):
    """Train a single model"""
    print("="*80)
    print(f"🎯 Training: {model_name.upper()}")
    print("="*80)
    
    # Import and run training script
    if model_name == 'segmentation':
        from train.train_segmentation import main as train_seg
        train_seg(
            data_root=DATASET_ROOT,
            checkpoint_dir=CHECKPOINT_DIR,
            epochs=EPOCHS_CONFIG[model_name],
            batch_size=BATCH_SIZE,
            device=DEVICE
        )
    
    elif model_name == 'pose':
        from train.train_pose import main as train_pose
        train_pose(
            data_root=DATASET_ROOT,
            checkpoint_dir=CHECKPOINT_DIR,
            epochs=EPOCHS_CONFIG[model_name],
            batch_size=BATCH_SIZE,
            device=DEVICE
        )
    
    elif model_name == 'warp':
        from train.train_warp import main as train_warp
        train_warp(
            data_root=DATASET_ROOT,
            checkpoint_dir=CHECKPOINT_DIR,
            epochs=EPOCHS_CONFIG[model_name],
            batch_size=BATCH_SIZE,
            device=DEVICE
        )
    
    elif model_name == 'tryon':
        from train.train_tryon import main as train_tryon
        train_tryon(
            data_root=DATASET_ROOT,
            checkpoint_dir=CHECKPOINT_DIR,
            epochs=EPOCHS_CONFIG[model_name],
            batch_size=BATCH_SIZE,
            device=DEVICE
        )
    
    elif model_name == 'refine':
        from train.train_refine import main as train_refine
        train_refine(
            data_root=DATASET_ROOT,
            checkpoint_dir=CHECKPOINT_DIR,
            epochs=EPOCHS_CONFIG[model_name],
            batch_size=BATCH_SIZE,
            device=DEVICE
        )
    
    print(f"\n✅ {model_name.upper()} training complete!\n")


def main():
    """Main training pipeline"""
    
    # Check dataset exists
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ Dataset not found at: {DATASET_ROOT}")
        print("\n📝 Update DATASET_ROOT in this script to point to your dataset")
        return
    
    # Check for train/test folders
    train_dir = Path(DATASET_ROOT) / 'train'
    test_dir = Path(DATASET_ROOT) / 'test'
    
    if not train_dir.exists():
        print(f"❌ Train folder not found: {train_dir}")
        return
    
    print(f"✅ Dataset structure validated")
    print(f"   Train: {train_dir}")
    print(f"   Test: {test_dir}")
    print()
    
    # Training order (dependencies considered)
    training_order = [
        ('segmentation', 'Segmentation U-Net'),
        ('pose', 'Pose Estimation'),
        ('warp', 'Cloth Warping TPS'),
        ('tryon', 'Try-On GAN'),
        ('refine', 'Refinement Network')
    ]
    
    print("📋 Training Pipeline:")
    for i, (name, desc) in enumerate(training_order, 1):
        print(f"   {i}. {desc} ({EPOCHS_CONFIG[name]} epochs)")
    print()
    
    # Ask for confirmation (skip in Kaggle)
    if not IS_KAGGLE:
        response = input("Start training? (yes/no): ").strip().lower()
        if response != 'yes':
            print("Training cancelled.")
            return
    
    # Train all models
    for model_name, model_desc in training_order:
        try:
            train_model(model_name, model_desc)
        except KeyboardInterrupt:
            print(f"\n⚠️  Training interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Ask if should continue
            if not IS_KAGGLE:
                response = input("\nContinue with next model? (yes/no): ").strip().lower()
                if response != 'yes':
                    break
            else:
                print("Continuing with next model...")
                continue
    
    print("\n" + "="*80)
    print("🎉 TRAINING PIPELINE COMPLETE!")
    print("="*80)
    print()
    print(f"📁 Checkpoints saved to: {CHECKPOINT_DIR}/")
    print()
    print("Next steps:")
    print("  1. Download checkpoints from Kaggle")
    print("  2. Run inference with trained models")
    print("  3. Evaluate results on test set")
    print()


if __name__ == '__main__':
    main()
