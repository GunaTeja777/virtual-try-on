"""
U-Net model for human parsing/segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double Convolution block"""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super(DoubleConv, self).__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Pad x1 to match x2 size
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution"""
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for human parsing
    
    Predicts segmentation masks with multiple classes:
    - Background
    - Hair
    - Face
    - Upper clothes
    - Lower clothes
    - Arms
    - Legs
    - etc.
    """
    def __init__(self, n_channels: int = 3, n_classes: int = 20, bilinear: bool = True):
        """
        Args:
            n_channels: Number of input channels (3 for RGB)
            n_classes: Number of segmentation classes
            bilinear: Use bilinear upsampling (True) or transposed conv (False)
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input image
        
        Returns:
            out: (B, n_classes, H, W) segmentation logits
        """
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits
    
    def predict(self, x):
        """
        Predict segmentation masks
        
        Args:
            x: (B, 3, H, W) input image
        
        Returns:
            masks: (B, H, W) predicted class indices
            probs: (B, n_classes, H, W) class probabilities
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        masks = torch.argmax(probs, dim=1)
        return masks, probs


class AttentionUNet(nn.Module):
    """
    U-Net with attention gates for better feature learning
    """
    def __init__(self, n_channels: int = 3, n_classes: int = 20):
        super(AttentionUNet, self).__init__()
        
        # Similar to U-Net but with attention mechanisms
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder
        self.enc1 = DoubleConv(n_channels, 64)
        self.enc2 = Down(64, 128)
        self.enc3 = Down(128, 256)
        self.enc4 = Down(256, 512)
        self.enc5 = Down(512, 1024)
        
        # Decoder with attention
        self.up1 = Up(1024, 512)
        self.att1 = AttentionGate(512, 512, 256)
        
        self.up2 = Up(512, 256)
        self.att2 = AttentionGate(256, 256, 128)
        
        self.up3 = Up(256, 128)
        self.att3 = AttentionGate(128, 128, 64)
        
        self.up4 = Up(128, 64)
        
        # Output
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        
        # Decoder with attention
        x = self.up1(x5, self.att1(x4, x5))
        x = self.up2(x, self.att2(x3, x))
        x = self.up3(x, self.att3(x2, x))
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits


class AttentionGate(nn.Module):
    """Attention gate module"""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        Args:
            g: gating signal from coarser scale
            x: skip connection from encoder
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


def get_segmentation_model(model_type: str = 'unet', n_classes: int = 20) -> nn.Module:
    """
    Factory function to get segmentation model
    
    Args:
        model_type: 'unet' or 'attention_unet'
        n_classes: Number of segmentation classes
    
    Returns:
        Segmentation model
    """
    if model_type == 'unet':
        return UNet(n_channels=3, n_classes=n_classes)
    elif model_type == 'attention_unet':
        return AttentionUNet(n_channels=3, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # Test U-Net
    print("Testing U-Net model...")
    
    model = UNet(n_channels=3, n_classes=20)
    x = torch.randn(2, 3, 256, 192)
    
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test prediction
    masks, probs = model.predict(x)
    print(f"Predicted masks shape: {masks.shape}")
    print(f"Probabilities shape: {probs.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params:,}")
    
    print("\n✅ U-Net model ready!")
