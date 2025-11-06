"""
Quick test to verify CUDA installation and start training
"""
import torch

print("="*60)
print("🔍 CUDA Setup Verification")
print("="*60)

print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("\n✅ GPU is ready for training!")
    print("="*60)
else:
    print("\n⚠️  CUDA not available. Training will use CPU.")
    print("   Make sure PyTorch with CUDA is installed:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("="*60)
