"""
Quick test script to verify installation and models
"""
import sys
import torch
import torchvision

print("="*60)
print("🔍 VIRTUAL TRY-ON SYSTEM - INSTALLATION TEST")
print("="*60)

# Test Python version
print(f"\n✓ Python version: {sys.version.split()[0]}")

# Test PyTorch
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ TorchVision version: {torchvision.__version__}")

# Test CUDA
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠ CUDA not available - will use CPU (slower)")

# Test imports
print("\n📦 Testing imports...")
try:
    from models.segmentation_unet import UNet
    print("✓ Segmentation model")
except Exception as e:
    print(f"✗ Segmentation model: {e}")

try:
    from models.pose_estimation import SimplePoseNet
    print("✓ Pose estimation model")
except Exception as e:
    print(f"✗ Pose estimation model: {e}")

try:
    from models.cloth_warp_tps import ClothWarpNet
    print("✓ Cloth warping model")
except Exception as e:
    print(f"✗ Cloth warping model: {e}")

try:
    from models.tryon_gan import TryOnGenerator
    print("✓ Try-on GAN")
except Exception as e:
    print(f"✗ Try-on GAN: {e}")

try:
    from models.refine_net import RefineNet
    print("✓ Refinement model")
except Exception as e:
    print(f"✗ Refinement model: {e}")

try:
    from utils.dataset_loader import VTONDataset
    print("✓ Dataset loader")
except Exception as e:
    print(f"✗ Dataset loader: {e}")

try:
    from utils.losses import CombinedLoss
    print("✓ Loss functions")
except Exception as e:
    print(f"✗ Loss functions: {e}")

# Test model creation
print("\n🧪 Testing model creation...")
try:
    model = UNet(n_channels=3, n_classes=20)
    x = torch.randn(1, 3, 256, 192)
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
    y = model(x)
    print(f"✓ U-Net forward pass: {x.shape} → {y.shape}")
except Exception as e:
    print(f"✗ Model test failed: {e}")

print("\n" + "="*60)
print("✅ INSTALLATION TEST COMPLETE!")
print("="*60)
print("\nNext steps:")
print("1. Prepare dataset: python main.py --raw_data_path raw_data")
print("2. Train models: python main.py --dataset_path dataset")
print("3. Run inference: python inference/test_pipeline.py --person <path> --cloth <path>")
print("4. Launch demo: streamlit run inference/demo_ui.py")
