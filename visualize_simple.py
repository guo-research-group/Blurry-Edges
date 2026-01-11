"""
Simple Side-by-Side Visualization
==================================
Creates easy-to-understand comparisons showing:
- Before (Sparse): What Blurry-Edges gives you
- After (Dense): What the densifier produces
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from models.depth_densifier import DepthDensifierUNet


def compute_boundary(depth):
    """Compute boundary map"""
    from scipy.ndimage import sobel
    dx = sobel(depth, axis=0)
    dy = sobel(depth, axis=1)
    boundary = np.sqrt(dx**2 + dy**2)
    boundary = (boundary - boundary.min()) / (boundary.max() - boundary.min() + 1e-8)
    return boundary


def load_model(checkpoint_path, device):
    """Load trained model"""
    model = DepthDensifierUNet(in_channels=6, out_channels=1)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def simple_comparison(idx, image, sparse_depth, dense_depth, gt_depth, save_path=None):
    """Create simple before/after comparison"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Compute masks
    sparse_mask = sparse_depth > 0
    gt_mask = gt_depth > 0
    
    # Stats
    sparse_coverage = sparse_mask.sum() / sparse_mask.size * 100
    if sparse_mask.sum() > 0 and gt_mask.sum() > 0:
        sparse_rmse = np.sqrt(((sparse_depth[sparse_mask & gt_mask] - gt_depth[sparse_mask & gt_mask]) ** 2).mean())
    else:
        sparse_rmse = np.nan
    
    if gt_mask.sum() > 0:
        dense_rmse = np.sqrt(((dense_depth[gt_mask] - gt_depth[gt_mask]) ** 2).mean())
    else:
        dense_rmse = np.nan
    
    # Depth range
    vmin = min(sparse_depth[sparse_mask].min() if sparse_mask.sum() > 0 else 0,
               dense_depth.min(),
               gt_depth[gt_mask].min() if gt_mask.sum() > 0 else 0)
    vmax = max(sparse_depth[sparse_mask].max() if sparse_mask.sum() > 0 else 1,
               dense_depth.max(),
               gt_depth[gt_mask].max() if gt_mask.sum() > 0 else 1)
    
    # Row 1: Input and depth maps
    # RGB
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Input RGB Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Sparse depth (BEFORE)
    sparse_vis = np.ma.masked_where(~sparse_mask, sparse_depth)
    im1 = axes[0, 1].imshow(sparse_vis, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'BEFORE: Sparse Depth\nCoverage: {sparse_coverage:.1f}%  |  RMSE: {sparse_rmse:.2f} cm',
                        fontsize=14, fontweight='bold', color='red', backgroundcolor='yellow')
    axes[0, 1].axis('off')
    # Add text overlay
    axes[0, 1].text(0.5, 0.95, 'Missing depth in many regions!', 
                   transform=axes[0, 1].transAxes,
                   fontsize=12, color='red', weight='bold',
                   ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Dense depth (AFTER)
    im2 = axes[0, 2].imshow(dense_depth, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f'AFTER: Dense Depth\nCoverage: 100%  |  RMSE: {dense_rmse:.2f} cm',
                        fontsize=14, fontweight='bold', color='green', backgroundcolor='lightgreen')
    axes[0, 2].axis('off')
    # Add text overlay
    axes[0, 2].text(0.5, 0.95, 'Complete depth everywhere!', 
                   transform=axes[0, 2].transAxes,
                   fontsize=12, color='green', weight='bold',
                   ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    # Row 2: Coverage and improvements
    # Coverage visualization
    coverage_before = np.zeros((*sparse_depth.shape, 3))
    coverage_before[sparse_mask] = [0, 1, 0]  # Green
    coverage_before[~sparse_mask] = [0.3, 0.3, 0.3]  # Dark gray (missing)
    axes[1, 0].imshow(coverage_before)
    axes[1, 0].set_title(f'BEFORE Coverage\n{sparse_coverage:.0f}% of pixels have depth',
                        fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    # Add annotation
    axes[1, 0].text(0.5, 0.05, 'Gray = Missing Depth', 
                   transform=axes[1, 0].transAxes,
                   fontsize=11, color='white', weight='bold',
                   ha='center', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
    
    # What was filled
    filled_regions = np.zeros((*sparse_depth.shape, 3))
    filled_regions[sparse_mask] = [0.5, 0.5, 0.5]  # Gray (already had)
    filled_regions[~sparse_mask] = [1, 1, 0]  # Yellow (newly filled)
    axes[1, 1].imshow(filled_regions)
    axes[1, 1].set_title(f'What Densifier Filled\n{100 - sparse_coverage:.0f}% pixels were filled',
                        fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    # Add annotation
    axes[1, 1].text(0.5, 0.05, 'Yellow = Newly Filled by Network', 
                   transform=axes[1, 1].transAxes,
                   fontsize=11, color='black', weight='bold',
                   ha='center', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))
    
    # Ground truth
    gt_vis = np.ma.masked_where(~gt_mask, gt_depth)
    im3 = axes[1, 2].imshow(gt_vis, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1, 2].set_title('Ground Truth\n(What we compare against)',
                        fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
    
    # Overall title
    improvement = ((sparse_rmse - dense_rmse) / sparse_rmse * 100) if not np.isnan(sparse_rmse) else 0
    if improvement > 0:
        result_text = f'✓ IMPROVEMENT: {improvement:.1f}% better RMSE'
        result_color = 'green'
    else:
        result_text = f'⚠ TRADE-OFF: {-improvement:.1f}% higher RMSE, but 100% coverage'
        result_color = 'orange'
    
    fig.suptitle(f'Image #{idx} - Before vs After Densification\n{result_text}',
                 fontsize=16, fontweight='bold', color=result_color)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  ✓ Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Simple Before/After Visualization')
    parser.add_argument('--data_path', type=str, default='./data_test/regular')
    parser.add_argument('--model_path', type=str, default='./pretrained_weights/best_densifier.pth')
    parser.add_argument('--start_idx', type=int, default=180)
    parser.add_argument('--num_images', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='./visualizations/simple')
    parser.add_argument('--cuda', type=str, default='cuda:0')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading data...")
    images = np.load(os.path.join(args.data_path, 'images_ny.npy'))
    gt_depths = np.load(os.path.join(args.data_path, 'depth_maps.npy'))
    
    # Load model
    print("Loading model...")
    model = load_model(args.model_path, device)
    print("Ready!\n")
    
    print("="*70)
    print("GENERATING SIMPLE BEFORE/AFTER COMPARISONS")
    print("="*70)
    
    for i in range(args.start_idx, min(args.start_idx + args.num_images, len(images))):
        print(f"\nImage #{i}...")
        
        # Load
        image = images[i, 0, :, :, :]
        if image.max() > 1.0:
            image = image / 255.0
        gt_depth = gt_depths[i]
        
        sparse_depth = np.load(f'./logs/blurry_edges_depths/depth_{i:03d}.npy')
        if len(sparse_depth.shape) == 3:
            sparse_depth = sparse_depth[0]
        
        confidence_path = f'./logs/blurry_edges_depths/confidence_{i:03d}.npy'
        if os.path.exists(confidence_path):
            confidence = np.load(confidence_path)
            if len(confidence.shape) == 3:
                confidence = confidence[0]
        else:
            confidence = np.ones_like(sparse_depth) * 0.5
        
        # Densify
        boundary = compute_boundary(sparse_depth)
        input_tensor = np.stack([
            sparse_depth, boundary, confidence,
            image[:, :, 0], image[:, :, 1], image[:, :, 2]
        ], axis=0)[np.newaxis, :, :, :].astype(np.float32)
        
        with torch.no_grad():
            dense_depth = model(torch.from_numpy(input_tensor).to(device))
        dense_depth = dense_depth.cpu().numpy()[0, 0]
        
        # Visualize
        save_path = os.path.join(args.output_dir, f'simple_{i:03d}.png')
        simple_comparison(i, image, sparse_depth, dense_depth, gt_depth, save_path)
    
    print("\n" + "="*70)
    print(f"✓ Done! Check: {args.output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
