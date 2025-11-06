"""
Pose Estimation Model using Stacked Hourglass Network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block"""
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualBlock, self).__init__()
        
        mid_channels = out_channels // 2
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = None
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.skip is not None:
            residual = self.skip(residual)
        
        out += residual
        out = self.relu(out)
        
        return out


class HourglassModule(nn.Module):
    """Single Hourglass module"""
    def __init__(self, depth: int, num_features: int):
        super(HourglassModule, self).__init__()
        self.depth = depth
        self.num_features = num_features
        
        self._generate_network(self.depth)
    
    def _generate_network(self, level):
        """Generate hourglass network recursively"""
        # Upper branch
        self.add_module(f'b1_{level}', ResidualBlock(self.num_features, self.num_features))
        
        # Lower branch
        self.add_module(f'b2_{level}', ResidualBlock(self.num_features, self.num_features))
        
        if level > 1:
            self._generate_network(level - 1)
        else:
            # Bottom of hourglass
            self.add_module(f'b2_plus_{level}', ResidualBlock(self.num_features, self.num_features))
        
        # Post-processing
        self.add_module(f'b3_{level}', ResidualBlock(self.num_features, self.num_features))
    
    def _forward(self, level, inp):
        """Forward through one level"""
        # Upper branch
        up1 = inp
        up1 = self._modules[f'b1_{level}'](up1)
        
        # Lower branch
        low1 = F.max_pool2d(inp, 2, stride=2)
        low1 = self._modules[f'b2_{level}'](low1)
        
        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules[f'b2_plus_{level}'](low2)
        
        low3 = low2
        low3 = self._modules[f'b3_{level}'](low3)
        
        # Upsample and add
        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')
        
        return up1 + up2
    
    def forward(self, x):
        return self._forward(self.depth, x)


class PoseEstimationModel(nn.Module):
    """
    Stacked Hourglass Network for pose estimation
    
    Predicts heatmaps for body keypoints
    """
    def __init__(self, num_stacks: int = 2, num_blocks: int = 1, 
                 num_classes: int = 18, num_features: int = 256):
        """
        Args:
            num_stacks: Number of stacked hourglasses
            num_blocks: Number of residual blocks per hourglass
            num_classes: Number of keypoints (18 for COCO)
            num_features: Number of features in hourglass
        """
        super(PoseEstimationModel, self).__init__()
        
        self.num_stacks = num_stacks
        self.num_classes = num_classes
        
        # Initial processing
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.res1 = ResidualBlock(64, 128)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.res2 = ResidualBlock(128, 128)
        self.res3 = ResidualBlock(128, num_features)
        
        # Stacked hourglasses
        self.hg = nn.ModuleList([
            nn.Sequential(*[HourglassModule(4, num_features) for _ in range(num_blocks)])
            for _ in range(num_stacks)
        ])
        
        self.res = nn.ModuleList([
            nn.Sequential(*[ResidualBlock(num_features, num_features) for _ in range(num_blocks)])
            for _ in range(num_stacks)
        ])
        
        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_features, num_features, kernel_size=1, bias=False),
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_stacks)
        ])
        
        self.score = nn.ModuleList([
            nn.Conv2d(num_features, num_classes, kernel_size=1)
            for _ in range(num_stacks)
        ])
        
        # For intermediate supervision
        if num_stacks > 1:
            self.fc_ = nn.ModuleList([
                nn.Conv2d(num_features, num_features, kernel_size=1)
                for _ in range(num_stacks - 1)
            ])
            
            self.score_ = nn.ModuleList([
                nn.Conv2d(num_classes, num_features, kernel_size=1)
                for _ in range(num_stacks - 1)
            ])
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input image
        
        Returns:
            out: List of (B, num_classes, H/4, W/4) heatmaps for each stack
        """
        # Initial processing
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.res1(x)
        x = self.pool(x)
        x = self.res2(x)
        x = self.res3(x)
        
        out = []
        
        for i in range(self.num_stacks):
            # Hourglass
            hg = self.hg[i](x)
            
            # Residual layers
            ll = self.res[i](hg)
            ll = self.fc[i](ll)
            
            # Predict heatmaps
            tmp_out = self.score[i](ll)
            out.append(tmp_out)
            
            # Prepare for next stack
            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](ll)
                score_ = self.score_[i](tmp_out)
                x = x + fc_ + score_
        
        return out
    
    def predict(self, x):
        """
        Predict keypoints from heatmaps
        
        Args:
            x: (B, 3, H, W) input image
        
        Returns:
            keypoints: (B, num_classes, 3) with (x, y, confidence)
        """
        # Get heatmaps from final stack
        heatmaps_list = self.forward(x)
        heatmaps = heatmaps_list[-1]  # (B, num_classes, H/4, W/4)
        
        batch_size, num_joints, h, w = heatmaps.shape
        
        # Resize heatmaps to original size
        heatmaps_resized = F.interpolate(heatmaps, size=(x.shape[2], x.shape[3]), 
                                         mode='bilinear', align_corners=True)
        
        # Find maximum locations
        heatmaps_flat = heatmaps_resized.view(batch_size, num_joints, -1)
        max_vals, max_indices = torch.max(heatmaps_flat, dim=2)
        
        # Convert to (x, y) coordinates
        keypoints = torch.zeros(batch_size, num_joints, 3, device=x.device)
        
        for b in range(batch_size):
            for j in range(num_joints):
                idx = max_indices[b, j]
                keypoints[b, j, 0] = idx % x.shape[3]  # x
                keypoints[b, j, 1] = idx // x.shape[3]  # y
                keypoints[b, j, 2] = max_vals[b, j]     # confidence
        
        return keypoints


class SimplePoseNet(nn.Module):
    """
    Simpler CNN-based pose estimation model (faster but less accurate)
    """
    def __init__(self, num_classes: int = 18):
        super(SimplePoseNet, self).__init__()
        
        self.num_classes = num_classes
        
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Output
        self.final_layer = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        
        Returns:
            heatmaps: (B, num_classes, H/4, W/4)
        """
        # Encode
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Decode
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        
        # Output heatmaps
        heatmaps = self.final_layer(x)
        
        return heatmaps


def get_pose_model(model_type: str = 'simple', num_classes: int = 18) -> nn.Module:
    """
    Factory function to get pose estimation model
    
    Args:
        model_type: 'simple' or 'hourglass'
        num_classes: Number of keypoints
    
    Returns:
        Pose estimation model
    """
    if model_type == 'simple':
        return SimplePoseNet(num_classes=num_classes)
    elif model_type == 'hourglass':
        return PoseEstimationModel(num_stacks=2, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # Test pose estimation models
    print("Testing Pose Estimation Models...")
    
    # Test simple model
    simple_model = SimplePoseNet(num_classes=18)
    x = torch.randn(2, 3, 256, 192)
    
    output = simple_model(x)
    print(f"\nSimple Model:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    
    num_params = sum(p.numel() for p in simple_model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")
    
    # Test hourglass model
    print(f"\nHourglass Model:")
    hourglass_model = PoseEstimationModel(num_stacks=2, num_classes=18)
    
    outputs = hourglass_model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Number of outputs: {len(outputs)}")
    print(f"  Output shape: {outputs[0].shape}")
    
    num_params = sum(p.numel() for p in hourglass_model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")
    
    print("\n✅ Pose estimation models ready!")
