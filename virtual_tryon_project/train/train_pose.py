"""
Training script for Pose Estimation Model
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

from models.pose_estimation import get_pose_model
from utils.dataset_loader import get_dataloader
from utils.visualization import plot_training_curves


def train_pose(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = get_pose_model(model_type=args.model_type, num_classes=args.num_keypoints)
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # DataLoader
    train_loader = get_dataloader(args.data_root, args.batch_size, 'train', 
                                  (args.img_height, args.img_width), args.num_workers)
    
    # Training
    best_loss = float('inf')
    losses = []
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            person_img = batch['person_image'].to(device)
            pose_heatmap = batch['pose_heatmap'].to(device)
            
            # Forward
            if args.model_type == 'hourglass':
                outputs = model(person_img)
                loss = sum([criterion(out, pose_heatmap) for out in outputs])
            else:
                output = model(person_img)
                loss = criterion(output, pose_heatmap)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
        
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 
                      os.path.join(args.checkpoint_dir, 'best_pose.pth'))
            print(f"✅ Saved best model: {best_loss:.4f}")
        
        scheduler.step()
    
    plot_training_curves({'loss': losses}, 
                        os.path.join(args.checkpoint_dir, 'pose_curves.png'))
    print("✅ Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../dataset')
    parser.add_argument('--checkpoint_dir', default='../checkpoints/pose')
    parser.add_argument('--model_type', default='simple', choices=['simple', 'hourglass'])
    parser.add_argument('--num_keypoints', type=int, default=18)
    parser.add_argument('--num_epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--img_height', type=int, default=256)
    parser.add_argument('--img_width', type=int, default=192)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train_pose(args)
