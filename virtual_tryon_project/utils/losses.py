"""
Loss functions for Virtual Try-On System
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    """
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) predicted probabilities
            target: (B, C, H, W) ground truth one-hot encoded
        """
        pred = pred.contiguous().view(pred.size(0), pred.size(1), -1)
        target = target.contiguous().view(target.size(0), target.size(1), -1)
        
        intersection = (pred * target).sum(dim=2)
        union = pred.sum(dim=2) + target.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features
    """
    def __init__(self, layers: list = None):
        super(PerceptualLoss, self).__init__()
        
        if layers is None:
            layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        
        self.layers = layers
        vgg = models.vgg19(pretrained=True).features
        self.vgg = vgg.eval()
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Layer mapping
        self.layer_name_mapping = {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '17': 'relu3_3',
            '26': 'relu4_3'
        }
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 3, H, W) predicted image
            target: (B, 3, H, W) target image
        """
        loss = 0.0
        
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        for layer in self.layers:
            loss += F.l1_loss(pred_features[layer], target_features[layer])
        
        return loss / len(self.layers)
    
    def extract_features(self, x: torch.Tensor) -> dict:
        """Extract features from VGG layers"""
        features = {}
        
        for name, module in self.vgg._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                layer_name = self.layer_name_mapping[name]
                if layer_name in self.layers:
                    features[layer_name] = x
        
        return features


class GANLoss(nn.Module):
    """
    GAN Loss (LSGAN by default)
    """
    def __init__(self, gan_mode: str = 'lsgan', target_real_label: float = 1.0, 
                 target_fake_label: float = 0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        
        self.gan_mode = gan_mode
        
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError(f'GAN mode {gan_mode} not implemented')
    
    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Create label tensor with same size as prediction"""
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        
        return target_tensor.expand_as(prediction)
    
    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """
        Args:
            prediction: discriminator output
            target_is_real: whether target is real or fake
        """
        if self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        else:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        
        return loss


class StyleLoss(nn.Module):
    """
    Style loss using Gram matrix
    """
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.perceptual = PerceptualLoss()
    
    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix"""
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 3, H, W)
            target: (B, 3, H, W)
        """
        pred_features = self.perceptual.extract_features(pred)
        target_features = self.perceptual.extract_features(target)
        
        loss = 0.0
        for layer in self.perceptual.layers:
            pred_gram = self.gram_matrix(pred_features[layer])
            target_gram = self.gram_matrix(target_features[layer])
            loss += F.mse_loss(pred_gram, target_gram)
        
        return loss / len(self.perceptual.layers)


class TVLoss(nn.Module):
    """
    Total Variation Loss for smoothness
    """
    def __init__(self, weight: float = 1.0):
        super(TVLoss, self).__init__()
        self.weight = weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        """
        batch_size, c, h, w = x.size()
        
        tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
        
        return self.weight * (tv_h + tv_w) / (batch_size * c * h * w)


class MaskConsistencyLoss(nn.Module):
    """
    Mask consistency loss for try-on
    """
    def __init__(self):
        super(MaskConsistencyLoss, self).__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 3, H, W) predicted image
            target: (B, 3, H, W) target image
            mask: (B, 1, H, W) mask region
        """
        masked_pred = pred * mask
        masked_target = target * mask
        
        loss = F.l1_loss(masked_pred, masked_target)
        return loss


class EdgeLoss(nn.Module):
    """
    Edge-aware loss using Sobel filter
    """
    def __init__(self):
        super(EdgeLoss, self).__init__()
        
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W)
            target: (B, C, H, W)
        """
        # Convert to grayscale if needed
        if pred.size(1) == 3:
            pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
            target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        else:
            pred_gray = pred
            target_gray = target
        
        # Compute edges
        pred_edge_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred_gray, self.sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_edge_x**2 + pred_edge_y**2)
        
        target_edge_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_edge_y = F.conv2d(target_gray, self.sobel_y, padding=1)
        target_edges = torch.sqrt(target_edge_x**2 + target_edge_y**2)
        
        return F.l1_loss(pred_edges, target_edges)


class CombinedLoss(nn.Module):
    """
    Combined loss for try-on training
    """
    def __init__(self, 
                 l1_weight: float = 1.0,
                 perceptual_weight: float = 0.1,
                 style_weight: float = 0.0,
                 tv_weight: float = 0.0001,
                 gan_weight: float = 1.0):
        super(CombinedLoss, self).__init__()
        
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.gan_weight = gan_weight
        
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()
        self.tv_loss = TVLoss()
        self.gan_loss = GANLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                disc_pred: torch.Tensor = None) -> dict:
        """
        Args:
            pred: predicted image
            target: target image
            disc_pred: discriminator prediction (optional)
        
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # L1 loss
        if self.l1_weight > 0:
            losses['l1'] = self.l1_loss(pred, target) * self.l1_weight
        
        # Perceptual loss
        if self.perceptual_weight > 0:
            losses['perceptual'] = self.perceptual_loss(pred, target) * self.perceptual_weight
        
        # Style loss
        if self.style_weight > 0:
            losses['style'] = self.style_loss(pred, target) * self.style_weight
        
        # TV loss
        if self.tv_weight > 0:
            losses['tv'] = self.tv_loss(pred) * self.tv_weight
        
        # GAN loss
        if disc_pred is not None and self.gan_weight > 0:
            losses['gan'] = self.gan_loss(disc_pred, True) * self.gan_weight
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


if __name__ == '__main__':
    # Test losses
    pred = torch.randn(2, 3, 256, 192)
    target = torch.randn(2, 3, 256, 192)
    
    combined_loss = CombinedLoss()
    losses = combined_loss(pred, target)
    
    print("Loss components:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
