"""
Training script for Refinement Model
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

from models.refine_net import get_refine_model
from utils.dataset_loader import get_dataloader
from utils.losses import EdgeLoss
from utils.visualization import plot_training_curves


def train_refine(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = get_refine_model(model_type=args.model_type)
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Losses
    l1_loss = nn.L1Loss()
    edge_loss = EdgeLoss().to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # DataLoader (assumes you have generated try-on images to refine)
    train_loader = get_dataloader(args.data_root, args.batch_size, 'train',
                                  (args.img_height, args.img_width), args.num_workers)
    
    # Training
    best_loss = float('inf')
    losses = {'total': [], 'l1': [], 'edge': []}
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = {'total': 0, 'l1': 0, 'edge': 0}
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            # Use person image as input (simulating coarse try-on result)
            coarse = batch['person_image'].to(device)
            target = batch['person_image'].to(device)  # Ground truth
            
            # Forward
            refined = model(coarse)
            
            # Losses
            loss_l1 = l1_loss(refined, target)
            loss_edge = edge_loss(refined, target)
            
            # Check for NaN/Inf
            if torch.isnan(loss_l1) or torch.isinf(loss_l1):
                print(f"⚠️  NaN/Inf in L1 loss! Skipping batch...")
                continue
            if torch.isnan(loss_edge) or torch.isinf(loss_edge):
                print(f"⚠️  NaN/Inf in Edge loss! Skipping batch...")
                continue
            
            total_loss = loss_l1 + 0.01 * loss_edge  # Reduced edge weight from 0.1 to 0.01
            
            # Backward with gradient clipping
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Metrics
            epoch_losses['total'] += total_loss.item()
            epoch_losses['l1'] += loss_l1.item()
            epoch_losses['edge'] += loss_edge.item()
        
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
            losses[key].append(epoch_losses[key])
        
        print(f"Epoch {epoch+1} - Loss: {epoch_losses['total']:.4f}")
        
        if epoch_losses['total'] < best_loss:
            best_loss = epoch_losses['total']
            torch.save(model.state_dict(),
                      os.path.join(args.checkpoint_dir, 'best_refine.pth'))
            print(f"✅ Saved: {best_loss:.4f}")
        
        scheduler.step()
    
    plot_training_curves(losses, os.path.join(args.checkpoint_dir, 'refine_curves.png'))
    print("✅ Completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../dataset')
    parser.add_argument('--checkpoint_dir', default='../checkpoints/refine')
    parser.add_argument('--model_type', default='standard', 
                       choices=['standard', 'lightweight', 'attention'])
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--img_height', type=int, default=256)
    parser.add_argument('--img_width', type=int, default=192)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train_refine(args)
