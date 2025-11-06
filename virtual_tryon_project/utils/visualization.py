"""
Visualization utilities for Virtual Try-On System
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List, Tuple, Optional
from pathlib import Path


def visualize_keypoints(image: np.ndarray, keypoints: np.ndarray, 
                       save_path: Optional[str] = None) -> np.ndarray:
    """
    Visualize pose keypoints on image
    
    Args:
        image: (H, W, 3) RGB image
        keypoints: (N, 3) array with (x, y, confidence)
        save_path: Optional path to save visualization
    
    Returns:
        vis_image: Image with keypoints drawn
    """
    vis_image = image.copy()
    
    # Define skeleton connections (COCO format)
    skeleton = [
        [0, 1], [0, 2], [1, 3], [2, 4],  # Head
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # Arms
        [11, 12], [5, 11], [6, 12],  # Torso
        [11, 13], [13, 15], [12, 14], [14, 16]  # Legs
    ]
    
    # Draw skeleton
    for connection in skeleton:
        pt1_idx, pt2_idx = connection
        if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
            pt1 = keypoints[pt1_idx]
            pt2 = keypoints[pt2_idx]
            
            if pt1[2] > 0 and pt2[2] > 0:  # Check confidence
                pt1_pos = (int(pt1[0]), int(pt1[1]))
                pt2_pos = (int(pt2[0]), int(pt2[1]))
                cv2.line(vis_image, pt1_pos, pt2_pos, (0, 255, 0), 2)
    
    # Draw keypoints
    for kp in keypoints:
        if kp[2] > 0:  # Check confidence
            cv2.circle(vis_image, (int(kp[0]), int(kp[1])), 4, (255, 0, 0), -1)
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    return vis_image


def visualize_segmentation(mask: np.ndarray, num_classes: int = 20,
                          save_path: Optional[str] = None) -> np.ndarray:
    """
    Visualize segmentation mask with colors
    
    Args:
        mask: (H, W) segmentation mask with class indices
        num_classes: Number of classes
        save_path: Optional path to save visualization
    
    Returns:
        colored_mask: (H, W, 3) colored segmentation
    """
    # Create color map
    colors = plt.cm.get_cmap('tab20', num_classes)
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    for class_id in range(num_classes):
        color = (np.array(colors(class_id)[:3]) * 255).astype(np.uint8)
        colored_mask[mask == class_id] = color
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
    
    return colored_mask


def visualize_tryon_results(person: np.ndarray, cloth: np.ndarray, 
                           result: np.ndarray, save_path: Optional[str] = None) -> np.ndarray:
    """
    Create side-by-side visualization of try-on results
    
    Args:
        person: (H, W, 3) person image
        cloth: (H, W, 3) cloth image
        result: (H, W, 3) try-on result
        save_path: Optional path to save visualization
    
    Returns:
        combined: (H, W*3, 3) combined visualization
    """
    # Ensure same height
    height = max(person.shape[0], cloth.shape[0], result.shape[0])
    
    # Resize if needed
    if person.shape[0] != height:
        person = cv2.resize(person, (person.shape[1], height))
    if cloth.shape[0] != height:
        cloth = cv2.resize(cloth, (cloth.shape[1], height))
    if result.shape[0] != height:
        result = cv2.resize(result, (result.shape[1], height))
    
    # Concatenate horizontally
    combined = np.concatenate([person, cloth, result], axis=1)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, 'Person', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, 'Cloth', (person.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, 'Try-On Result', (person.shape[1] + cloth.shape[1] + 10, 30), 
                font, 1, (255, 255, 255), 2)
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    
    return combined


def visualize_warping(cloth: np.ndarray, warped_cloth: np.ndarray,
                     save_path: Optional[str] = None) -> np.ndarray:
    """
    Visualize cloth warping results
    
    Args:
        cloth: (H, W, 3) original cloth
        warped_cloth: (H, W, 3) warped cloth
        save_path: Optional path to save visualization
    
    Returns:
        combined: (H, W*2, 3) combined visualization
    """
    combined = np.concatenate([cloth, warped_cloth], axis=1)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, 'Original Cloth', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, 'Warped Cloth', (cloth.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    
    return combined


def plot_training_curves(losses: dict, save_path: str):
    """
    Plot training loss curves
    
    Args:
        losses: Dictionary with loss names as keys and lists of values
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 6))
    
    for loss_name, loss_values in losses.items():
        plt.plot(loss_values, label=loss_name)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_heatmaps(heatmaps: torch.Tensor, save_path: Optional[str] = None) -> np.ndarray:
    """
    Visualize pose heatmaps
    
    Args:
        heatmaps: (N, H, W) heatmaps for N joints
        save_path: Optional path to save visualization
    
    Returns:
        vis_heatmap: Visualization of all heatmaps
    """
    num_joints = heatmaps.shape[0]
    
    # Convert to numpy
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.cpu().numpy()
    
    # Create grid visualization
    grid_size = int(np.ceil(np.sqrt(num_joints)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    for i in range(num_joints):
        axes[i].imshow(heatmaps[i], cmap='hot')
        axes[i].set_title(f'Joint {i}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_joints, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
    
    return fig


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array for visualization
    
    Args:
        tensor: (C, H, W) or (B, C, H, W) tensor
    
    Returns:
        array: (H, W, C) or (B, H, W, C) numpy array
    """
    if tensor.dim() == 4:
        # Batch of images
        array = tensor.permute(0, 2, 3, 1).cpu().numpy()
    elif tensor.dim() == 3:
        # Single image
        array = tensor.permute(1, 2, 0).cpu().numpy()
    else:
        array = tensor.cpu().numpy()
    
    # Denormalize if needed (assumes [-1, 1] range)
    if array.min() < 0:
        array = (array + 1.0) * 127.5
    
    array = np.clip(array, 0, 255).astype(np.uint8)
    
    return array


def create_grid_visualization(images: List[torch.Tensor], 
                              titles: List[str] = None,
                              save_path: Optional[str] = None) -> np.ndarray:
    """
    Create grid visualization of multiple images
    
    Args:
        images: List of tensors (C, H, W)
        titles: List of titles for each image
        save_path: Optional path to save visualization
    
    Returns:
        grid_image: Grid visualization
    """
    num_images = len(images)
    cols = min(4, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if num_images == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, img in enumerate(images):
        img_np = tensor_to_numpy(img)
        axes[i].imshow(img_np)
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return fig


def save_checkpoint_visualization(epoch: int, outputs: dict, save_dir: str):
    """
    Save visualization of training outputs at checkpoint
    
    Args:
        epoch: Current epoch
        outputs: Dictionary containing tensors to visualize
        save_dir: Directory to save visualizations
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualization for each output
    images = []
    titles = []
    
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor) and value.dim() >= 3:
            # Take first sample from batch
            img = value[0] if value.dim() == 4 else value
            images.append(img)
            titles.append(key)
    
    if images:
        save_path = save_dir / f'epoch_{epoch:04d}.png'
        create_grid_visualization(images, titles, str(save_path))


if __name__ == '__main__':
    # Test visualization functions
    print("Testing visualization utilities...")
    
    # Test keypoints
    image = np.random.randint(0, 255, (256, 192, 3), dtype=np.uint8)
    keypoints = np.random.rand(17, 3) * np.array([192, 256, 1])
    vis_kp = visualize_keypoints(image, keypoints)
    print("✅ Keypoint visualization works!")
    
    # Test segmentation
    mask = np.random.randint(0, 10, (256, 192))
    vis_seg = visualize_segmentation(mask, num_classes=10)
    print("✅ Segmentation visualization works!")
    
    print("\n✅ All visualization utilities ready!")
