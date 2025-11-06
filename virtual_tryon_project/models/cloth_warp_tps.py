"""
Cloth Warping Model using TPS transformation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')
from utils.tps_transform import TPSTransform, extract_control_points_from_keypoints


class ClothEncoder(nn.Module):
    """Encoder for cloth image"""
    def __init__(self, in_channels: int = 3, out_features: int = 256):
        super(ClothEncoder, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, out_features)
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) cloth image
        
        Returns:
            features: (B, out_features) cloth features
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        features = self.fc(x)
        
        return features


class PoseEncoder(nn.Module):
    """Encoder for pose representation"""
    def __init__(self, in_channels: int = 18, out_features: int = 256):
        super(PoseEncoder, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, out_features)
    
    def forward(self, x):
        """
        Args:
            x: (B, 18, H, W) pose heatmaps
        
        Returns:
            features: (B, out_features) pose features
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        features = self.fc(x)
        
        return features


class ControlPointPredictor(nn.Module):
    """Predict TPS control point displacements"""
    def __init__(self, feature_dim: int = 512, num_control_points: int = 25):
        super(ControlPointPredictor, self).__init__()
        
        self.num_control_points = num_control_points
        
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_control_points * 2)  # (x, y) for each point
        )
        
        # Initialize to predict small displacements
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.zero_()
    
    def forward(self, features):
        """
        Args:
            features: (B, feature_dim) combined features
        
        Returns:
            displacements: (B, num_control_points, 2) control point displacements
        """
        displacements = self.fc(features)
        displacements = displacements.view(-1, self.num_control_points, 2)
        
        # Limit displacement range
        displacements = torch.tanh(displacements) * 0.5
        
        return displacements


class ClothWarpNet(nn.Module):
    """
    Cloth Warping Network using TPS transformation
    
    Takes cloth image and pose, predicts warped cloth
    """
    def __init__(self, num_control_points: int = 25):
        super(ClothWarpNet, self).__init__()
        
        self.num_control_points = num_control_points
        
        # Encoders
        self.cloth_encoder = ClothEncoder(in_channels=3, out_features=256)
        self.pose_encoder = PoseEncoder(in_channels=18, out_features=256)
        
        # Control point predictor
        self.control_point_predictor = ControlPointPredictor(
            feature_dim=512,
            num_control_points=num_control_points
        )
        
        # TPS transformation
        self.tps = TPSTransform(grid_size=int(num_control_points**0.5))
    
    def forward(self, cloth, pose_heatmap):
        """
        Args:
            cloth: (B, 3, H, W) cloth image
            pose_heatmap: (B, 18, H, W) pose heatmaps
        
        Returns:
            warped_cloth: (B, 3, H, W) warped cloth image
            source_points: (B, N, 2) source control points
            target_points: (B, N, 2) target control points
        """
        batch_size = cloth.size(0)
        
        # Extract features
        cloth_features = self.cloth_encoder(cloth)
        pose_features = self.pose_encoder(pose_heatmap)
        
        # Combine features
        combined_features = torch.cat([cloth_features, pose_features], dim=1)
        
        # Predict control point displacements
        displacements = self.control_point_predictor(combined_features)
        
        # Create source control points (regular grid)
        source_points = self.tps.control_points.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Target control points = source + displacement
        target_points = source_points + displacements
        
        # Apply TPS transformation
        warped_cloth = self.tps(cloth, source_points, target_points)
        
        return warped_cloth, source_points, target_points


class FlowBasedWarpNet(nn.Module):
    """
    Alternative: Flow-based warping (predicts dense flow field)
    """
    def __init__(self):
        super(FlowBasedWarpNet, self).__init__()
        
        # Encoder for cloth + pose
        self.encoder = nn.Sequential(
            nn.Conv2d(3 + 18, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Decoder to predict flow
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 2, kernel_size=7, stride=1, padding=3)
        )
    
    def forward(self, cloth, pose_heatmap):
        """
        Args:
            cloth: (B, 3, H, W)
            pose_heatmap: (B, 18, H, W)
        
        Returns:
            warped_cloth: (B, 3, H, W)
            flow: (B, 2, H, W)
        """
        # Concatenate inputs
        x = torch.cat([cloth, pose_heatmap], dim=1)
        
        # Predict flow
        features = self.encoder(x)
        flow = self.decoder(features)
        
        # Warp cloth using flow
        warped_cloth = self.warp_with_flow(cloth, flow)
        
        return warped_cloth, flow
    
    def warp_with_flow(self, image, flow):
        """Warp image using optical flow"""
        batch_size, _, height, width = image.shape
        
        # Create base grid
        y = torch.linspace(-1, 1, height, device=image.device)
        x = torch.linspace(-1, 1, width, device=image.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=0)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        # Normalize flow
        flow_norm = flow / torch.tensor([width, height], device=flow.device).view(1, 2, 1, 1) * 2
        
        # Add flow to grid
        warped_grid = grid + flow_norm
        warped_grid = warped_grid.permute(0, 2, 3, 1)
        
        # Warp image
        warped_image = F.grid_sample(image, warped_grid, 
                                     mode='bilinear',
                                     padding_mode='border',
                                     align_corners=True)
        
        return warped_image


def get_warp_model(model_type: str = 'tps', **kwargs) -> nn.Module:
    """
    Factory function to get warping model
    
    Args:
        model_type: 'tps' or 'flow'
    
    Returns:
        Warping model
    """
    if model_type == 'tps':
        return ClothWarpNet(**kwargs)
    elif model_type == 'flow':
        return FlowBasedWarpNet()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # Test cloth warping models
    print("Testing Cloth Warping Models...")
    
    batch_size = 2
    cloth = torch.randn(batch_size, 3, 256, 192)
    pose = torch.randn(batch_size, 18, 256, 192)
    
    # Test TPS-based warping
    print("\nTPS-based Warping:")
    tps_model = ClothWarpNet(num_control_points=25)
    warped, source_pts, target_pts = tps_model(cloth, pose)
    
    print(f"  Input cloth shape: {cloth.shape}")
    print(f"  Input pose shape: {pose.shape}")
    print(f"  Warped cloth shape: {warped.shape}")
    print(f"  Source points shape: {source_pts.shape}")
    print(f"  Target points shape: {target_pts.shape}")
    
    num_params = sum(p.numel() for p in tps_model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")
    
    # Test flow-based warping
    print("\nFlow-based Warping:")
    flow_model = FlowBasedWarpNet()
    warped_flow, flow = flow_model(cloth, pose)
    
    print(f"  Warped cloth shape: {warped_flow.shape}")
    print(f"  Flow shape: {flow.shape}")
    
    num_params = sum(p.numel() for p in flow_model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")
    
    print("\n✅ Cloth warping models ready!")
