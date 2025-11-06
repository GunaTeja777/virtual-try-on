"""
Virtual Try-On Inference Script
Uses trained Segmentation and Try-On GAN models
"""
import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.segmentation_unet import get_segmentation_model
from models.tryon_gan import get_tryon_models


class VirtualTryOnSystem:
    def __init__(self, 
                 segmentation_ckpt='checkpoints/segmentation/best_segmentation.pth',
                 generator_ckpt='checkpoints/tryon/best_generator.pth',
                 device='cuda'):
        """
        Initialize Virtual Try-On System
        
        Args:
            segmentation_ckpt: Path to segmentation model checkpoint
            generator_ckpt: Path to generator checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🎮 Using device: {self.device}")
        
        # Load Segmentation Model
        print("📦 Loading Segmentation Model...")
        self.seg_model = get_segmentation_model(model_type='unet', n_classes=20)
        checkpoint = torch.load(segmentation_ckpt, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.seg_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.seg_model.load_state_dict(checkpoint)
        self.seg_model = self.seg_model.to(self.device)
        self.seg_model.eval()
        print("   ✅ Segmentation model loaded")
        
        # Load Try-On GAN Generator
        print("📦 Loading Try-On Generator...")
        self.generator, _ = get_tryon_models()
        self.generator.load_state_dict(torch.load(generator_ckpt, map_location=self.device))
        self.generator = self.generator.to(self.device)
        self.generator.eval()
        print("   ✅ Generator loaded")
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 192)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print("✅ Virtual Try-On System Ready!\n")
    
    def load_image(self, image_path):
        """Load and preprocess image"""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0)
        return img_tensor, img
    
    def segment_person(self, person_tensor):
        """Generate segmentation mask for person"""
        with torch.no_grad():
            person_tensor = person_tensor.to(self.device)
            seg_logits = self.seg_model(person_tensor)
            seg_mask = torch.argmax(seg_logits, dim=1, keepdim=True).float()
            # Normalize to [0, 1]
            seg_mask = seg_mask / 20.0
        return seg_mask
    
    def create_dummy_pose(self, batch_size=1):
        """Create dummy pose heatmap (since we have dummy pose data)"""
        # 18 keypoints, all zeros
        pose = torch.zeros(batch_size, 18, 256, 192)
        return pose.to(self.device)
    
    def try_on(self, person_image_path, cloth_image_path):
        """
        Perform virtual try-on
        
        Args:
            person_image_path: Path to person image
            cloth_image_path: Path to cloth image
            
        Returns:
            result_image: PIL Image of try-on result
            intermediate: Dictionary with intermediate results
        """
        print(f"🎨 Processing Try-On...")
        print(f"   Person: {person_image_path}")
        print(f"   Cloth: {cloth_image_path}")
        
        # Load images
        person_tensor, person_img = self.load_image(person_image_path)
        cloth_tensor, cloth_img = self.load_image(cloth_image_path)
        
        # Generate segmentation
        print("   → Generating segmentation...")
        seg_mask = self.segment_person(person_tensor)
        
        # Create pose (dummy)
        pose = self.create_dummy_pose()
        
        # Move to device
        person_tensor = person_tensor.to(self.device)
        cloth_tensor = cloth_tensor.to(self.device)
        
        # Generate try-on result
        print("   → Generating try-on result...")
        with torch.no_grad():
            # Generator expects separate inputs: warped_cloth, segmentation, pose_heatmap
            # Since we don't have warping trained, we use cloth directly
            result_tensor = self.generator(cloth_tensor, seg_mask, pose)
        
        # Convert to image
        result_tensor = result_tensor.cpu()
        result_np = result_tensor[0].permute(1, 2, 0).numpy()
        result_np = (result_np * 0.5 + 0.5) * 255  # Denormalize from [-1, 1] to [0, 255]
        result_np = np.clip(result_np, 0, 255).astype(np.uint8)
        result_image = Image.fromarray(result_np)
        
        print("   ✅ Try-on complete!\n")
        
        # Prepare intermediate results for visualization
        seg_np = seg_mask[0, 0].cpu().numpy()
        seg_np = (seg_np * 255).astype(np.uint8)
        seg_img = Image.fromarray(seg_np).convert('RGB')
        
        intermediate = {
            'person': person_img,
            'cloth': cloth_img,
            'segmentation': seg_img,
            'result': result_image
        }
        
        return result_image, intermediate
    
    def save_results(self, intermediate, output_path):
        """Save results with visualization"""
        # Create a grid of images
        person = intermediate['person'].resize((192, 256))
        cloth = intermediate['cloth'].resize((192, 256))
        seg = intermediate['segmentation'].resize((192, 256))
        result = intermediate['result'].resize((192, 256))
        
        # Create composite image
        width = 192 * 4 + 30  # 4 images with padding
        height = 256 + 40  # Image height + padding
        
        composite = Image.new('RGB', (width, height), (255, 255, 255))
        
        # Paste images
        composite.paste(person, (10, 20))
        composite.paste(cloth, (202, 20))
        composite.paste(seg, (394, 20))
        composite.paste(result, (586, 20))
        
        # Save
        composite.save(output_path)
        print(f"💾 Results saved to: {output_path}")
        
        return composite


def demo():
    """Run demo with sample images from dataset"""
    print("="*70)
    print("🎯 Virtual Try-On System - Demo")
    print("="*70)
    print()
    
    # Initialize system
    system = VirtualTryOnSystem(
        segmentation_ckpt='checkpoints/segmentation/best_segmentation.pth',
        generator_ckpt='checkpoints/tryon/best_generator.pth',
        device='cuda'
    )
    
    # Get sample images from dataset
    dataset_path = Path('dataset')
    person_dir = dataset_path / 'person'
    cloth_dir = dataset_path / 'cloth_organized'
    
    # Find first available person and cloth
    person_files = sorted(list(person_dir.glob('*.jpg')))
    cloth_files = sorted(list(cloth_dir.glob('*.jpg')))
    
    if len(person_files) == 0 or len(cloth_files) == 0:
        print("❌ Error: No images found in dataset!")
        return
    
    print(f"📁 Found {len(person_files)} person images")
    print(f"📁 Found {len(cloth_files)} cloth images\n")
    
    # Create output directory
    output_dir = Path('inference_results')
    output_dir.mkdir(exist_ok=True)
    
    # Test with first 3 combinations
    num_tests = min(3, len(person_files), len(cloth_files))
    
    for i in range(num_tests):
        person_path = person_files[i]
        cloth_path = cloth_files[i]
        
        print(f"Test {i+1}/{num_tests}:")
        
        # Run try-on
        result, intermediate = system.try_on(str(person_path), str(cloth_path))
        
        # Save results
        output_path = output_dir / f'tryon_result_{i+1}.png'
        system.save_results(intermediate, output_path)
        
        print()
    
    print("="*70)
    print(f"✅ Demo Complete! Check '{output_dir}/' for results")
    print("="*70)


def inference_single(person_path, cloth_path, output_path=None):
    """
    Run inference on a single person-cloth pair
    
    Args:
        person_path: Path to person image
        cloth_path: Path to cloth image
        output_path: Path to save result (optional)
    """
    # Initialize system
    system = VirtualTryOnSystem()
    
    # Run try-on
    result, intermediate = system.try_on(person_path, cloth_path)
    
    # Save or display
    if output_path:
        system.save_results(intermediate, output_path)
    else:
        result.show()
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Virtual Try-On Inference')
    parser.add_argument('--demo', action='store_true', help='Run demo with dataset samples')
    parser.add_argument('--person', type=str, help='Path to person image')
    parser.add_argument('--cloth', type=str, help='Path to cloth image')
    parser.add_argument('--output', type=str, help='Path to save result')
    
    args = parser.parse_args()
    
    if args.demo:
        demo()
    elif args.person and args.cloth:
        inference_single(args.person, args.cloth, args.output)
    else:
        print("Usage:")
        print("  Demo mode: python inference.py --demo")
        print("  Single inference: python inference.py --person <path> --cloth <path> --output <path>")
