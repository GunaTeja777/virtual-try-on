"""
Thin-Plate Spline (TPS) Transformation for cloth warping
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TPSTransform(nn.Module):
    """
    Thin-Plate Spline transformation for spatial warping
    """
    def __init__(self, grid_size: int = 5):
        """
        Args:
            grid_size: Number of control points per dimension
        """
        super(TPSTransform, self).__init__()
        self.grid_size = grid_size
        
        # Create regular grid of control points
        self.register_buffer('control_points', self._create_control_grid(grid_size))
    
    def _create_control_grid(self, grid_size: int) -> torch.Tensor:
        """Create regular grid of control points"""
        x = torch.linspace(-1, 1, grid_size)
        y = torch.linspace(-1, 1, grid_size)
        
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        control_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        
        return control_points  # (grid_size^2, 2)
    
    def _compute_distance_matrix(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances between source and target points
        
        Args:
            source: (B, N, 2)
            target: (B, M, 2)
        
        Returns:
            distances: (B, N, M)
        """
        # Expand dimensions for broadcasting
        source_expanded = source.unsqueeze(2)  # (B, N, 1, 2)
        target_expanded = target.unsqueeze(1)  # (B, 1, M, 2)
        
        # Compute Euclidean distance
        diff = source_expanded - target_expanded
        distances = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
        
        return distances
    
    def _compute_tps_kernel(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Compute TPS radial basis function: U(r) = r^2 * log(r)
        
        Args:
            distances: (B, N, M)
        
        Returns:
            kernel: (B, N, M)
        """
        # Avoid log(0)
        kernel = distances ** 2 * torch.log(distances + 1e-8)
        return kernel
    
    def _solve_tps(self, source_points: torch.Tensor, 
                   target_points: torch.Tensor) -> torch.Tensor:
        """
        Solve TPS interpolation problem
        
        Args:
            source_points: (B, N, 2) control points in source
            target_points: (B, N, 2) control points in target
        
        Returns:
            weights: (B, N+3, 2) TPS weights
        """
        batch_size, num_points, _ = source_points.shape
        
        # Compute kernel matrix K
        distances = self._compute_distance_matrix(source_points, source_points)
        K = self._compute_tps_kernel(distances)  # (B, N, N)
        
        # Create P matrix [1, x, y]
        ones = torch.ones(batch_size, num_points, 1, device=source_points.device)
        P = torch.cat([ones, source_points], dim=2)  # (B, N, 3)
        
        # Construct full matrix
        # | K  P |
        # | P^T 0 |
        zeros = torch.zeros(batch_size, 3, 3, device=source_points.device)
        top = torch.cat([K, P], dim=2)  # (B, N, N+3)
        bottom = torch.cat([P.transpose(1, 2), zeros], dim=2)  # (B, 3, N+3)
        L = torch.cat([top, bottom], dim=1)  # (B, N+3, N+3)
        
        # Create target vector
        zeros_target = torch.zeros(batch_size, 3, 2, device=target_points.device)
        V = torch.cat([target_points, zeros_target], dim=1)  # (B, N+3, 2)
        
        # Solve linear system: L * weights = V
        weights = torch.linalg.solve(L, V)  # (B, N+3, 2)
        
        return weights
    
    def _apply_tps(self, weights: torch.Tensor, source_points: torch.Tensor,
                   grid: torch.Tensor) -> torch.Tensor:
        """
        Apply TPS transformation to grid
        
        Args:
            weights: (B, N+3, 2) TPS weights
            source_points: (B, N, 2) control points
            grid: (B, H, W, 2) sampling grid
        
        Returns:
            warped_grid: (B, H, W, 2)
        """
        batch_size, height, width, _ = grid.shape
        num_points = source_points.shape[1]
        
        # Reshape grid to (B, H*W, 2)
        grid_flat = grid.view(batch_size, -1, 2)
        
        # Compute distances from grid points to control points
        distances = self._compute_distance_matrix(grid_flat, source_points)  # (B, H*W, N)
        
        # Compute kernel values
        K = self._compute_tps_kernel(distances)  # (B, H*W, N)
        
        # Create P matrix
        ones = torch.ones(batch_size, height * width, 1, device=grid.device)
        P = torch.cat([ones, grid_flat], dim=2)  # (B, H*W, 3)
        
        # Combine [K, P]
        full_basis = torch.cat([K, P], dim=2)  # (B, H*W, N+3)
        
        # Apply transformation
        warped_flat = torch.bmm(full_basis, weights)  # (B, H*W, 2)
        
        # Reshape back to grid
        warped_grid = warped_flat.view(batch_size, height, width, 2)
        
        return warped_grid
    
    def forward(self, image: torch.Tensor, source_points: torch.Tensor, 
                target_points: torch.Tensor) -> torch.Tensor:
        """
        Warp image using TPS transformation
        
        Args:
            image: (B, C, H, W) input image
            source_points: (B, N, 2) control points in source space [-1, 1]
            target_points: (B, N, 2) control points in target space [-1, 1]
        
        Returns:
            warped_image: (B, C, H, W)
        """
        batch_size, channels, height, width = image.shape
        
        # Solve TPS system
        weights = self._solve_tps(source_points, target_points)
        
        # Create sampling grid
        y = torch.linspace(-1, 1, height, device=image.device)
        x = torch.linspace(-1, 1, width, device=image.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=2)  # (H, W, 2)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (B, H, W, 2)
        
        # Apply TPS transformation
        warped_grid = self._apply_tps(weights, source_points, grid)
        
        # Sample from image using grid_sample
        warped_image = F.grid_sample(image, warped_grid, 
                                     mode='bilinear', 
                                     padding_mode='border',
                                     align_corners=True)
        
        return warped_image


class FlowWarp(nn.Module):
    """
    Warp image using optical flow field
    """
    def __init__(self):
        super(FlowWarp, self).__init__()
    
    def forward(self, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp image using flow field
        
        Args:
            image: (B, C, H, W) input image
            flow: (B, 2, H, W) flow field (dx, dy)
        
        Returns:
            warped_image: (B, C, H, W)
        """
        batch_size, _, height, width = image.shape
        
        # Create base grid
        y = torch.linspace(-1, 1, height, device=image.device)
        x = torch.linspace(-1, 1, width, device=image.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=0)  # (2, H, W)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (B, 2, H, W)
        
        # Add flow to grid
        # Flow should be normalized to [-1, 1] range
        flow_norm = flow / torch.tensor([width, height], device=flow.device).view(1, 2, 1, 1) * 2
        warped_grid = grid + flow_norm
        
        # Rearrange to (B, H, W, 2) for grid_sample
        warped_grid = warped_grid.permute(0, 2, 3, 1)
        
        # Warp image
        warped_image = F.grid_sample(image, warped_grid,
                                     mode='bilinear',
                                     padding_mode='border',
                                     align_corners=True)
        
        return warped_image


def extract_control_points_from_keypoints(keypoints: torch.Tensor, 
                                         img_size: tuple) -> torch.Tensor:
    """
    Extract control points from pose keypoints
    
    Args:
        keypoints: (B, N, 3) pose keypoints (x, y, confidence)
        img_size: (height, width)
    
    Returns:
        control_points: (B, N, 2) normalized to [-1, 1]
    """
    height, width = img_size
    
    # Extract x, y coordinates
    points = keypoints[:, :, :2]  # (B, N, 2)
    
    # Normalize to [-1, 1]
    points[:, :, 0] = (points[:, :, 0] / width) * 2 - 1
    points[:, :, 1] = (points[:, :, 1] / height) * 2 - 1
    
    return points


if __name__ == '__main__':
    # Test TPS transformation
    batch_size = 2
    channels = 3
    height, width = 256, 192
    
    # Create dummy image
    image = torch.randn(batch_size, channels, height, width)
    
    # Create control points
    num_points = 10
    source_points = torch.rand(batch_size, num_points, 2) * 2 - 1
    target_points = source_points + torch.randn(batch_size, num_points, 2) * 0.1
    
    # Apply TPS
    tps = TPSTransform()
    warped = tps(image, source_points, target_points)
    
    print(f"Input shape: {image.shape}")
    print(f"Warped shape: {warped.shape}")
    print("✅ TPS transformation successful!")
