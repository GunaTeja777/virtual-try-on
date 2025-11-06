"""
Complete Virtual Try-On Inference Pipeline
"""
import os
import sys
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import argparse

sys.path.append('..')
from models.segmentation_unet import get_segmentation_model
from models.pose_estimation import get_pose_model
from models.cloth_warp_tps import get_warp_model
from models.tryon_gan import get_tryon_models
from models.refine_net import get_refine_model
from utils.visualization import visualize_tryon_results, tensor_to_numpy


class VirtualTryOnPipeline:
    """
    End-to-end Virtual Try-On pipeline
    """
    def __init__(self, checkpoint_dir: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = Path(checkpoint_dir)
        
        print("🚀 Loading models...")
        
        # Load segmentation model
        self.segmentation_model = get_segmentation_model('unet', n_classes=20)
        self._load_checkpoint(self.segmentation_model, 
                            self.checkpoint_dir / 'segmentation' / 'best_segmentation.pth')
        self.segmentation_model = self.segmentation_model.to(self.device)
        self.segmentation_model.eval()
        print("✅ Segmentation model loaded")
        
        # Load pose estimation model
        self.pose_model = get_pose_model('simple', num_classes=18)
        self._load_checkpoint(self.pose_model,
                            self.checkpoint_dir / 'pose' / 'best_pose.pth')
        self.pose_model = self.pose_model.to(self.device)
        self.pose_model.eval()
        print("✅ Pose model loaded")
        
        # Load cloth warping model
        self.warp_model = get_warp_model('tps', num_control_points=25)
        self._load_checkpoint(self.warp_model,
                            self.checkpoint_dir / 'warp' / 'best_warp.pth')
        self.warp_model = self.warp_model.to(self.device)
        self.warp_model.eval()
        print("✅ Warp model loaded")
        
        # Load try-on GAN
        self.generator, _ = get_tryon_models('unet', 'patchgan')
        self._load_checkpoint(self.generator,
                            self.checkpoint_dir / 'tryon' / 'best_generator.pth')
        self.generator = self.generator.to(self.device)
        self.generator.eval()
        print("✅ Generator loaded")
        
        # Load refinement model
        self.refine_model = get_refine_model('standard')
        refine_path = self.checkpoint_dir / 'refine' / 'best_refine.pth'
        if refine_path.exists():
            self._load_checkpoint(self.refine_model, refine_path)
            self.refine_model = self.refine_model.to(self.device)
            self.refine_model.eval()
            print("✅ Refinement model loaded")
        else:
            self.refine_model = None
            print("⚠️  Refinement model not found, skipping")
        
        print("\n✅ All models loaded successfully!\n")
    
    def _load_checkpoint(self, model, checkpoint_path):
        """Load model checkpoint"""
        if not checkpoint_path.exists():
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    def preprocess_image(self, image_path: str, img_size: tuple = (256, 192)) -> torch.Tensor:
        """Load and preprocess image"""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size[1], img_size[0]))
        
        # Normalize to [-1, 1]
        img = (img.astype(np.float32) / 127.5) - 1.0
        
        # To tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img_tensor
    
    @torch.no_grad()
    def __call__(self, person_image_path: str, cloth_image_path: str, 
                 output_path: str = None) -> np.ndarray:
        """
        Run complete try-on pipeline
        
        Args:
            person_image_path: Path to person image
            cloth_image_path: Path to cloth image
            output_path: Optional path to save result
        
        Returns:
            result_image: Try-on result as numpy array
        """
        print("📸 Step 1: Loading images...")
        person_img = self.preprocess_image(person_image_path).to(self.device)
        cloth_img = self.preprocess_image(cloth_image_path).to(self.device)
        
        print("🎭 Step 2: Human parsing (segmentation)...")
        segmentation_logits = self.segmentation_model(person_img)
        segmentation = torch.argmax(segmentation_logits, dim=1, keepdim=True).float()
        segmentation = segmentation / 20.0  # Normalize
        
        print("🦴 Step 3: Pose estimation...")
        pose_heatmap = self.pose_model(person_img)
        if isinstance(pose_heatmap, list):
            pose_heatmap = pose_heatmap[-1]
        
        # Resize pose heatmap if needed
        if pose_heatmap.shape[2:] != person_img.shape[2:]:
            pose_heatmap = F.interpolate(pose_heatmap, size=person_img.shape[2:],
                                        mode='bilinear', align_corners=True)
        
        print("👕 Step 4: Cloth warping...")
        warped_cloth, _, _ = self.warp_model(cloth_img, pose_heatmap)
        
        print("🎨 Step 5: Try-on synthesis...")
        tryon_result = self.generator(warped_cloth, segmentation, pose_heatmap)
        
        print("✨ Step 6: Refinement...")
        if self.refine_model is not None:
            final_result = self.refine_model(tryon_result)
        else:
            final_result = tryon_result
        
        # Convert to numpy
        result_np = tensor_to_numpy(final_result[0])
        person_np = tensor_to_numpy(person_img[0])
        cloth_np = tensor_to_numpy(cloth_img[0])
        
        # Create visualization
        combined = visualize_tryon_results(person_np, cloth_np, result_np)
        
        # Save if requested
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            print(f"💾 Saved result to: {output_path}")
        
        print("✅ Try-on completed!\n")
        return result_np


def main():
    parser = argparse.ArgumentParser(description='Virtual Try-On Inference')
    parser.add_argument('--person', type=str, required=True, help='Path to person image')
    parser.add_argument('--cloth', type=str, required=True, help='Path to cloth image')
    parser.add_argument('--output', type=str, default='result.jpg', help='Output path')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = VirtualTryOnPipeline(args.checkpoint_dir, args.device)
    
    # Run try-on
    result = pipeline(args.person, args.cloth, args.output)
    
    print("🎉 Virtual Try-On successful!")


if __name__ == '__main__':
    main()
