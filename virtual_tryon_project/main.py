"""
Main training script - One-click training for all models
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_command(command, description):
    """Run a command with description"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error in: {description}")
        print(f"Command: {command}")
        return False
    
    print(f"\n✅ Completed: {description}")
    return True


def preprocess_dataset(args):
    """Preprocess raw dataset"""
    print("\n" + "="*60)
    print("📦 STEP 0: Dataset Preprocessing")
    print("="*60)
    
    sys.path.append('.')
    from utils.preprocessing import DatasetPreprocessor
    
    if not Path(args.raw_data_path).exists():
        print(f"⚠️  Raw data path not found: {args.raw_data_path}")
        print("Please place your dataset ZIP files in this directory:")
        print("  - images.zip")
        print("  - keypoints.zip")
        print("  - segm.zip")
        print("  - (optional) densepose.zip, labels.zip, captions.json")
        return False
    
    preprocessor = DatasetPreprocessor(
        raw_data_path=args.raw_data_path,
        output_path=args.dataset_path,
        img_size=(args.img_height, args.img_width)
    )
    
    preprocessor.run_preprocessing()
    
    return True


def train_all_models(args):
    """Train all models in sequence"""
    
    # Training configuration
    common_args = f"--data_root {args.dataset_path} " \
                 f"--batch_size {args.batch_size} " \
                 f"--img_height {args.img_height} " \
                 f"--img_width {args.img_width} " \
                 f"--num_workers {args.num_workers}"
    
    training_steps = [
        {
            'script': 'train/train_segmentation.py',
            'description': 'Training Segmentation Model (U-Net)',
            'args': f"{common_args} --num_epochs {args.seg_epochs} " \
                   f"--checkpoint_dir checkpoints/segmentation"
        },
        {
            'script': 'train/train_pose.py',
            'description': 'Training Pose Estimation Model',
            'args': f"{common_args} --num_epochs {args.pose_epochs} " \
                   f"--checkpoint_dir checkpoints/pose"
        },
        {
            'script': 'train/train_warp.py',
            'description': 'Training Cloth Warping Model (TPS)',
            'args': f"{common_args} --num_epochs {args.warp_epochs} " \
                   f"--checkpoint_dir checkpoints/warp"
        },
        {
            'script': 'train/train_tryon.py',
            'description': 'Training Try-On GAN',
            'args': f"{common_args} --num_epochs {args.tryon_epochs} " \
                   f"--checkpoint_dir checkpoints/tryon"
        },
        {
            'script': 'train/train_refine.py',
            'description': 'Training Refinement Model',
            'args': f"{common_args} --num_epochs {args.refine_epochs} " \
                   f"--checkpoint_dir checkpoints/refine"
        }
    ]
    
    # Execute training steps
    for step in training_steps:
        if args.skip_preprocessing and step['description'].startswith('Training'):
            # Check if checkpoint exists
            checkpoint_dir = step['args'].split('--checkpoint_dir')[-1].strip().split()[0]
            if Path(checkpoint_dir).exists() and list(Path(checkpoint_dir).glob('best_*.pth')):
                print(f"\n✅ Checkpoint found for {step['description']}, skipping...")
                continue
        
        command = f"python {step['script']} {step['args']}"
        success = run_command(command, step['description'])
        
        if not success and not args.continue_on_error:
            print("\n❌ Training stopped due to error")
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Virtual Try-On: One-Click Training')
    
    # Paths
    parser.add_argument('--raw_data_path', type=str, default='raw_data',
                       help='Path to raw dataset (ZIP files)')
    parser.add_argument('--dataset_path', type=str, default='dataset',
                       help='Path to preprocessed dataset')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--img_height', type=int, default=256,
                       help='Image height')
    parser.add_argument('--img_width', type=int, default=192,
                       help='Image width')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Epochs for each model
    parser.add_argument('--seg_epochs', type=int, default=100,
                       help='Segmentation training epochs')
    parser.add_argument('--pose_epochs', type=int, default=80,
                       help='Pose estimation training epochs')
    parser.add_argument('--warp_epochs', type=int, default=60,
                       help='Cloth warping training epochs')
    parser.add_argument('--tryon_epochs', type=int, default=100,
                       help='Try-on GAN training epochs')
    parser.add_argument('--refine_epochs', type=int, default=50,
                       help='Refinement training epochs')
    
    # Options
    parser.add_argument('--skip_preprocessing', action='store_true',
                       help='Skip dataset preprocessing')
    parser.add_argument('--continue_on_error', action='store_true',
                       help='Continue training even if a step fails')
    parser.add_argument('--train_only', type=str, default=None,
                       choices=['segmentation', 'pose', 'warp', 'tryon', 'refine'],
                       help='Train only specific model')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎨 VIRTUAL TRY-ON SYSTEM - ONE-CLICK TRAINING")
    print("="*60)
    
    # Step 0: Preprocess dataset
    if not args.skip_preprocessing:
        success = preprocess_dataset(args)
        if not success:
            print("\n⚠️  Preprocessing failed or skipped")
            print("You can skip preprocessing with --skip_preprocessing flag")
            return
    else:
        print("\n⏭️  Skipping preprocessing (--skip_preprocessing flag)")
    
    # Check dataset exists
    if not Path(args.dataset_path).exists():
        print(f"\n❌ Dataset not found: {args.dataset_path}")
        print("Please run preprocessing first or check the path")
        return
    
    # Train models
    print("\n" + "="*60)
    print("🏋️ STARTING MODEL TRAINING")
    print("="*60)
    print(f"\nDataset: {args.dataset_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_height}x{args.img_width}")
    print(f"\nTraining order:")
    print("  1️⃣  Segmentation U-Net")
    print("  2️⃣  Pose Estimation")
    print("  3️⃣  Cloth Warping (TPS)")
    print("  4️⃣  Try-On GAN")
    print("  5️⃣  Refinement Network")
    print("\n⏱️  This will take several hours depending on your hardware...")
    
    input("\nPress Enter to start training (or Ctrl+C to cancel)...")
    
    success = train_all_models(args)
    
    if success:
        print("\n" + "="*60)
        print("🎉 ALL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nCheckpoints saved in: checkpoints/")
        print("\nNext steps:")
        print("  1. Test the pipeline: python inference/test_pipeline.py --person <path> --cloth <path>")
        print("  2. Launch demo UI: streamlit run inference/demo_ui.py")
        print("\n✅ Your Virtual Try-On system is ready!")
    else:
        print("\n❌ Training completed with errors")
        print("Check the logs above for details")


if __name__ == '__main__':
    main()
