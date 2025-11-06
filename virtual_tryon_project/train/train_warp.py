"""
Training script for Cloth Warping Model
"""
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.cloth_warp_tps import get_warp_model
from utils.dataset_loader import get_dataloader
from utils.losses import PerceptualLoss, TVLoss
from utils.visualization import plot_training_curves


def train_warp(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = get_warp_model(model_type=args.model_type, num_control_points=args.num_control_points)
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Losses
    l1_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss().to(device)
    tv_loss = TVLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    # DataLoader
    train_loader = get_dataloader(args.data_root, args.batch_size, 'train',
                                  (args.img_height, args.img_width), args.num_workers)
    
    # Training
    best_loss = float('inf')
    losses = {'total': [], 'l1': [], 'perceptual': [], 'tv': []}
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = {'total': 0, 'l1': 0, 'perceptual': 0, 'tv': 0}
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            cloth = batch['cloth_image'].to(device)
            pose = batch['pose_heatmap'].to(device)
            person = batch['person_image'].to(device)  # Target for warping
            
            # Forward
            if args.model_type == 'tps':
                warped_cloth, _, _ = model(cloth, pose)
            else:
                warped_cloth, _ = model(cloth, pose)
            
            # Losses
            loss_l1 = l1_loss(warped_cloth, person)
            loss_perceptual = perceptual_loss(warped_cloth, person)
            loss_tv = tv_loss(warped_cloth)
            
            total_loss = loss_l1 + 0.1 * loss_perceptual + 0.001 * loss_tv
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Metrics
            epoch_losses['total'] += total_loss.item()
            epoch_losses['l1'] += loss_l1.item()
            epoch_losses['perceptual'] += loss_perceptual.item()
            epoch_losses['tv'] += loss_tv.item()
        
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
            losses[key].append(epoch_losses[key])
        
        print(f"Epoch {epoch+1} - Loss: {epoch_losses['total']:.4f}")
        
        if epoch_losses['total'] < best_loss:
            best_loss = epoch_losses['total']
            torch.save(model.state_dict(),
                      os.path.join(args.checkpoint_dir, 'best_warp.pth'))
            print(f"✅ Saved: {best_loss:.4f}")
        
        scheduler.step()
    
    plot_training_curves(losses, os.path.join(args.checkpoint_dir, 'warp_curves.png'))
    print("✅ Completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../dataset')
    parser.add_argument('--checkpoint_dir', default='../checkpoints/warp')
    parser.add_argument('--model_type', default='tps', choices=['tps', 'flow'])
    parser.add_argument('--num_control_points', type=int, default=25)
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--img_height', type=int, default=256)
    parser.add_argument('--img_width', type=int, default=192)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train_warp(args)
