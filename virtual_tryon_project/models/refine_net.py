"""
Refinement Network for enhancing try-on results
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RefineBlock(nn.Module):
    """Refinement block with dilated convolutions"""
    def __init__(self, channels: int, dilation: int = 1):
        super(RefineBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, 
                              padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                              padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + residual
        out = self.relu(out)
        
        return out


class RefineNet(nn.Module):
    """
    Refinement Network to enhance generated try-on images
    
    Focuses on:
    - Edge refinement
    - Texture enhancement
    - Color correction
    - Artifact removal
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super(RefineNet, self).__init__()
        
        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale refinement blocks with different dilation rates
        self.refine1 = RefineBlock(base_channels, dilation=1)
        self.refine2 = RefineBlock(base_channels, dilation=2)
        self.refine3 = RefineBlock(base_channels, dilation=4)
        self.refine4 = RefineBlock(base_channels, dilation=8)
        
        # Edge enhancement branch
        self.edge_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, base_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # Texture enhancement branch
        self.texture_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, base_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # Fusion (edge_feat: 32 + texture_feat: 32 = 64 channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final refinement
        self.final = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, kernel_size=7, padding=3),
            nn.Tanh()
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) generated try-on image
        
        Returns:
            refined: (B, 3, H, W) refined image
        """
        # Initial features
        feat = self.initial(x)
        
        # Multi-scale refinement
        feat = self.refine1(feat)
        feat = self.refine2(feat)
        feat = self.refine3(feat)
        feat = self.refine4(feat)
        
        # Edge and texture branches
        edge_feat = self.edge_conv(feat)
        texture_feat = self.texture_conv(feat)
        
        # Combine features
        combined = torch.cat([edge_feat, texture_feat], dim=1)
        fused = self.fusion(combined)
        
        # Final refinement
        refined = self.final(fused)
        
        # Residual connection with input
        refined = refined + x
        refined = torch.tanh(refined)  # Ensure output in [-1, 1]
        
        return refined


class LightweightRefineNet(nn.Module):
    """
    Lightweight refinement network for faster inference
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super(LightweightRefineNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks
        self.res1 = RefineBlock(base_channels * 2, dilation=1)
        self.res2 = RefineBlock(base_channels * 2, dilation=2)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels, in_channels, kernel_size=5, padding=2),
            nn.Tanh()
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        
        Returns:
            refined: (B, 3, H, W)
        """
        feat = self.encoder(x)
        feat = self.res1(feat)
        feat = self.res2(feat)
        refined = self.decoder(feat)
        
        # Residual connection
        refined = refined + x
        refined = torch.tanh(refined)
        
        return refined


class AttentionRefineNet(nn.Module):
    """
    Refinement network with attention mechanism
    Focuses on problematic regions
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super(AttentionRefineNet, self).__init__()
        
        # Feature extraction
        self.feat_extract = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Refinement blocks
        self.refine1 = RefineBlock(base_channels, dilation=1)
        self.refine2 = RefineBlock(base_channels, dilation=2)
        
        # Attention module
        self.attention = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, kernel_size=7, padding=3),
            nn.Tanh()
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        
        Returns:
            refined: (B, 3, H, W)
        """
        # Extract features
        feat = self.feat_extract(x)
        
        # Refine
        feat = self.refine1(feat)
        feat = self.refine2(feat)
        
        # Compute attention map
        attention_map = self.attention(feat)
        
        # Apply attention
        feat = feat * attention_map
        
        # Generate output
        refined = self.output(feat)
        
        # Residual connection
        refined = refined + x
        refined = torch.tanh(refined)
        
        return refined


def get_refine_model(model_type: str = 'standard', **kwargs) -> nn.Module:
    """
    Factory function to get refinement model
    
    Args:
        model_type: 'standard', 'lightweight', or 'attention'
    
    Returns:
        Refinement model
    """
    if model_type == 'standard':
        return RefineNet(**kwargs)
    elif model_type == 'lightweight':
        return LightweightRefineNet(**kwargs)
    elif model_type == 'attention':
        return AttentionRefineNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # Test refinement models
    print("Testing Refinement Models...")
    
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 192)
    
    # Test standard RefineNet
    print("\nStandard RefineNet:")
    model_std = RefineNet(in_channels=3, base_channels=64)
    output = model_std(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    num_params = sum(p.numel() for p in model_std.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")
    
    # Test lightweight RefineNet
    print("\nLightweight RefineNet:")
    model_light = LightweightRefineNet(in_channels=3, base_channels=32)
    output = model_light(x)
    print(f"  Output shape: {output.shape}")
    num_params = sum(p.numel() for p in model_light.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")
    
    # Test attention RefineNet
    print("\nAttention RefineNet:")
    model_att = AttentionRefineNet(in_channels=3, base_channels=64)
    output = model_att(x)
    print(f"  Output shape: {output.shape}")
    num_params = sum(p.numel() for p in model_att.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")
    
    print("\n✅ Refinement models ready!")
