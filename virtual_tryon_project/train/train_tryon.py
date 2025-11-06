"""
Training script for Try-On GAN
"""
import os
import sys
from pathlib import Path
import torch
import torch.optim as optim
from tqdm import tqdm
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.tryon_gan import get_tryon_models
from utils.dataset_loader import get_dataloader
from utils.losses import CombinedLoss, GANLoss
from utils.visualization import plot_training_curves


def train_tryon(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Models
    generator, discriminator = get_tryon_models(
        generator_type=args.gen_type,
        discriminator_type=args.disc_type
    )
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    print(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Losses
    combined_loss = CombinedLoss(l1_weight=1.0, perceptual_weight=0.1, 
                                 gan_weight=0.5).to(device)
    gan_loss = GANLoss(gan_mode='lsgan').to(device)
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # DataLoader
    train_loader = get_dataloader(args.data_root, args.batch_size, 'train',
                                  (args.img_height, args.img_width), args.num_workers)
    
    # Training
    losses = {'G': [], 'D': []}
    best_g_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        epoch_losses = {'G': 0, 'D': 0}
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            person = batch['person_image'].to(device)
            cloth = batch['cloth_image'].to(device)
            segm = batch['segmentation'].to(device)
            pose = batch['pose_heatmap'].to(device)
            
            # ===== Train Generator =====
            optimizer_G.zero_grad()
            
            # Generate try-on image
            fake_tryon = generator(cloth, segm, pose)
            
            # Discriminator prediction on fake
            pred_fake = discriminator(person, fake_tryon)
            
            # Generator losses
            g_losses = combined_loss(fake_tryon, person, pred_fake)
            g_loss = g_losses['total']
            
            g_loss.backward()
            optimizer_G.step()
            
            # ===== Train Discriminator =====
            optimizer_D.zero_grad()
            
            # Real images
            pred_real = discriminator(person, person)
            loss_d_real = gan_loss(pred_real, True)
            
            # Fake images
            pred_fake = discriminator(person, fake_tryon.detach())
            loss_d_fake = gan_loss(pred_fake, False)
            
            # Total discriminator loss
            d_loss = (loss_d_real + loss_d_fake) * 0.5
            
            d_loss.backward()
            optimizer_D.step()
            
            # Metrics
            epoch_losses['G'] += g_loss.item()
            epoch_losses['D'] += d_loss.item()
        
        # Average
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
            losses[key].append(epoch_losses[key])
        
        print(f"Epoch {epoch+1} - G: {epoch_losses['G']:.4f}, D: {epoch_losses['D']:.4f}")
        
        # Save
        if epoch_losses['G'] < best_g_loss:
            best_g_loss = epoch_losses['G']
            torch.save(generator.state_dict(),
                      os.path.join(args.checkpoint_dir, 'best_generator.pth'))
            torch.save(discriminator.state_dict(),
                      os.path.join(args.checkpoint_dir, 'best_discriminator.pth'))
            print(f"✅ Saved best: {best_g_loss:.4f}")
    
    plot_training_curves(losses, os.path.join(args.checkpoint_dir, 'tryon_curves.png'))
    print("✅ Completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../dataset')
    parser.add_argument('--checkpoint_dir', default='../checkpoints/tryon')
    parser.add_argument('--gen_type', default='unet', choices=['resnet', 'unet'])
    parser.add_argument('--disc_type', default='patchgan', choices=['patchgan', 'multiscale'])
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--img_height', type=int, default=256)
    parser.add_argument('--img_width', type=int, default=192)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train_tryon(args)
