"""
Training script for Human Parsing/Segmentation Model
"""
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.segmentation_unet import get_segmentation_model
from utils.dataset_loader import get_dataloader
from utils.losses import DiceLoss
from utils.visualization import save_checkpoint_visualization, plot_training_curves


def train_segmentation(args):
    """Train segmentation model"""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = get_segmentation_model(model_type=args.model_type, n_classes=args.num_classes)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss functions
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # DataLoader
    train_loader = get_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        mode='train',
        img_size=(args.img_height, args.img_width),
        num_workers=args.num_workers
    )
    
    # Training loop
    best_loss = float('inf')
    losses_history = {'total': [], 'ce': [], 'dice': []}
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = {'total': 0, 'ce': 0, 'dice': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            person_img = batch['person_image'].to(device)
            segmentation = batch['segmentation'].to(device)
            
            # Forward pass
            logits = model(person_img)
            
            # Create one-hot encoded target
            target_indices = segmentation.squeeze(1).long()
            target_onehot = torch.zeros(logits.size(), device=device)
            target_onehot.scatter_(1, target_indices.unsqueeze(1), 1)
            
            # Compute losses
            loss_ce = ce_loss(logits, target_indices)
            loss_dice = dice_loss(torch.softmax(logits, dim=1), target_onehot)
            total_loss = loss_ce + loss_dice
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_losses['total'] += total_loss.item()
            epoch_losses['ce'] += loss_ce.item()
            epoch_losses['dice'] += loss_dice.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss.item(),
                'ce': loss_ce.item(),
                'dice': loss_dice.item()
            })
        
        # Average epoch losses
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
            losses_history[key].append(epoch_losses[key])
        
        print(f"Epoch {epoch+1} - Loss: {epoch_losses['total']:.4f}")
        
        # Save checkpoint
        if epoch_losses['total'] < best_loss:
            best_loss = epoch_losses['total']
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_segmentation.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, checkpoint_path)
            print(f"✅ Saved best model with loss: {best_loss:.4f}")
        
        # Regular checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'segmentation_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
        
        scheduler.step()
    
    # Plot training curves
    plot_path = os.path.join(args.checkpoint_dir, 'training_curves.png')
    plot_training_curves(losses_history, plot_path)
    
    print("\n✅ Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Human Parsing Model')
    parser.add_argument('--data_root', type=str, default='../dataset', help='Dataset root directory')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints/segmentation', help='Checkpoint directory')
    parser.add_argument('--model_type', type=str, default='unet', choices=['unet', 'attention_unet'], help='Model type')
    parser.add_argument('--num_classes', type=int, default=20, help='Number of segmentation classes')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--img_height', type=int, default=256, help='Image height')
    parser.add_argument('--img_width', type=int, default=192, help='Image width')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save_freq', type=int, default=10, help='Save checkpoint frequency')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    train_segmentation(args)
