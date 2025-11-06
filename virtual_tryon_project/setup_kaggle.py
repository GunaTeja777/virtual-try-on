"""
Kaggle Environment Setup Script
Run this FIRST in your Kaggle notebook to fix compatibility issues
"""
import subprocess
import sys

def run_command(cmd, description):
    """Run shell command and handle errors"""
    print(f"\n{'='*70}")
    print(f"🔧 {description}")
    print(f"{'='*70}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Success!")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"⚠️  Warning: {result.stderr}")
    
    return result.returncode == 0


def setup_kaggle_environment():
    """Setup Kaggle environment with correct package versions"""
    
    print("="*70)
    print("🚀 KAGGLE ENVIRONMENT SETUP")
    print("="*70)
    print()
    
    # Step 1: Fix NumPy version
    print("Step 1: Fixing NumPy compatibility issue...")
    run_command("pip uninstall -y numpy", "Uninstalling NumPy 2.x")
    run_command("pip install 'numpy<2.0'", "Installing NumPy 1.x")
    
    # Step 2: Install core dependencies
    print("\nStep 2: Installing dependencies...")
    packages = [
        "albumentations",
        "opencv-python-headless",
        "scikit-image",
        "tqdm",
        "matplotlib",
        "scipy"
    ]
    
    for pkg in packages:
        run_command(f"pip install -q {pkg}", f"Installing {pkg}")
    
    # Step 3: Verify installations
    print("\n" + "="*70)
    print("✅ VERIFICATION")
    print("="*70)
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
        
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        import albumentations as A
        print(f"✅ Albumentations: {A.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  No GPU detected!")
        
        print("\n" + "="*70)
        print("🎉 Environment setup complete!")
        print("="*70)
        print()
        print("Next: Run the training script:")
        print("   !python train_all_kaggle.py")
        print()
        
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        print("Please check the error messages above.")


if __name__ == '__main__':
    setup_kaggle_environment()
