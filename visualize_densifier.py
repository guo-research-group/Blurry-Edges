"""
Visualization Script for Depth Densifier
==========================================
Creates side-by-side comparisons of:
1. Input RGB image
2. Sparse depth (Blurry-Edges baseline)
3. Dense depth (Densifier output)
4. Ground truth depth
5. Error maps

This helps understand what the densifier does visually.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

from models.depth_densifier import DepthDensifierUNet


def compute_boundary(depth):
    """Compute boundary map from depth"""
    from scipy.ndimage import sobel
    dx = sobel(depth, axis=0)
    dy = sobel(depth, axis=1)
    boundary = np.sqrt(dx**2 + dy**2)
    boundary = (boundary - boundary.min()) / (boundary.max() - boundary.min() + 1e-8)
    return boundary


def load_model(checkpoint_path, device):
    """Load trained densifier model"""
    model = DepthDensifierUNet(in_channels=6, out_channels=1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def visualize_comparison(idx, image, sparse_depth, confidence, dense_depth, gt_depth, save_path=None):
    """
    Create comprehensive visualization comparing sparse vs dense depth
    
    Args:
        idx: Image index
        image: RGB image [H, W, 3]
        sparse_depth: Sparse depth from Blurry-Edges [H, W]
        confidence: Confidence map [H, W]
        dense_depth: Dense depth from densifier [H, W]
        gt_depth: Ground truth depth [H, W]
        save_path: Path to save figure (optional)
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Color map for depth
    cmap = 'viridis'
    
    # Compute valid masks
    sparse_mask = sparse_depth > 0
    gt_mask = gt_depth > 0
    
    # Compute coverage
    sparse_coverage = sparse_mask.sum() / sparse_mask.size * 100
    dense_coverage = 100.0  # Always 100%
    
    # Compute errors (only where GT exists)
    sparse_error = np.abs(sparse_depth - gt_depth)
    sparse_error[~sparse_mask] = 0
    dense_error = np.abs(dense_depth - gt_depth)
    
    # Compute metrics
    if sparse_mask.sum() > 0:
        sparse_rmse = np.sqrt((sparse_error[sparse_mask & gt_mask] ** 2).mean())
        sparse_mae = sparse_error[sparse_mask & gt_mask].mean()
    else:
        sparse_rmse = np.nan
        sparse_mae = np.nan
    
    if gt_mask.sum() > 0:
        dense_rmse = np.sqrt((dense_error[gt_mask] ** 2).mean())
        dense_mae = dense_error[gt_mask].mean()
    else:
        dense_rmse = np.nan
        dense_mae = np.nan
    
    # Row 1: Input and confidence
    # 1. RGB Image
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(image)
    ax1.set_title('(a) Input RGB Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Confidence Map
    ax2 = plt.subplot(3, 4, 2)
    im2 = ax2.imshow(confidence, cmap='hot', vmin=0, vmax=1)
    ax2.set_title('(b) Confidence Map\n(Blurry-Edges)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Confidence')
    
    # 3. Coverage Visualization
    ax3 = plt.subplot(3, 4, 3)
    coverage_viz = np.zeros((*sparse_depth.shape, 3))
    coverage_viz[sparse_mask] = [0, 1, 0]  # Green = has depth
    coverage_viz[~sparse_mask] = [1, 0, 0]  # Red = no depth
    ax3.imshow(coverage_viz)
    ax3.set_title(f'(c) Sparse Coverage\n{sparse_coverage:.1f}% pixels', 
                  fontsize=12, fontweight='bold')
    ax3.axis('off')
    # Add legend
    green_patch = mpatches.Patch(color='green', label='Has depth')
    red_patch = mpatches.Patch(color='red', label='Missing depth')
    ax3.legend(handles=[green_patch, red_patch], loc='upper right', fontsize=8)
    
    # 4. Dense Coverage (always 100%)
    ax4 = plt.subplot(3, 4, 4)
    coverage_viz_dense = np.zeros((*dense_depth.shape, 3))
    coverage_viz_dense[:] = [0, 1, 0]  # All green
    ax4.imshow(coverage_viz_dense)
    ax4.set_title(f'(d) Dense Coverage\n{dense_coverage:.1f}% pixels', 
                  fontsize=12, fontweight='bold')
    ax4.axis('off')
    # Add legend
    ax4.legend(handles=[green_patch], loc='upper right', fontsize=8)
    
    # Row 2: Depth maps
    # Determine common depth range for consistent coloring
    valid_depths = [sparse_depth[sparse_mask], dense_depth, gt_depth[gt_mask]]
    valid_depths = [d for d in valid_depths if len(d) > 0]
    if len(valid_depths) > 0:
        vmin = min([d.min() for d in valid_depths])
        vmax = max([d.max() for d in valid_depths])
    else:
        vmin, vmax = 0, 1
    
    # 5. Sparse Depth (Blurry-Edges)
    ax5 = plt.subplot(3, 4, 5)
    sparse_vis = np.ma.masked_where(~sparse_mask, sparse_depth)
    im5 = ax5.imshow(sparse_vis, cmap=cmap, vmin=vmin, vmax=vmax)
    ax5.set_title(f'(e) Sparse Depth (Baseline)\nRMSE: {sparse_rmse:.2f} cm', 
                  fontsize=12, fontweight='bold', color='blue')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04, label='Depth (cm)')
    
    # 6. Dense Depth (Densifier)
    ax6 = plt.subplot(3, 4, 6)
    im6 = ax6.imshow(dense_depth, cmap=cmap, vmin=vmin, vmax=vmax)
    ax6.set_title(f'(f) Dense Depth (Ours)\nRMSE: {dense_rmse:.2f} cm', 
                  fontsize=12, fontweight='bold', color='green')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04, label='Depth (cm)')
    
    # 7. Ground Truth
    ax7 = plt.subplot(3, 4, 7)
    gt_vis = np.ma.masked_where(~gt_mask, gt_depth)
    im7 = ax7.imshow(gt_vis, cmap=cmap, vmin=vmin, vmax=vmax)
    ax7.set_title('(g) Ground Truth Depth', fontsize=12, fontweight='bold')
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04, label='Depth (cm)')
    
    # 8. Difference (Dense - Sparse) - shows what was filled
    ax8 = plt.subplot(3, 4, 8)
    diff = np.zeros_like(dense_depth)
    diff[sparse_mask] = dense_depth[sparse_mask] - sparse_depth[sparse_mask]
    diff[~sparse_mask] = dense_depth[~sparse_mask]  # Newly filled regions
    im8 = ax8.imshow(diff, cmap='RdBu_r', vmin=-5, vmax=5)
    ax8.set_title('(h) Filled Regions\n(Red=newly filled)', fontsize=12, fontweight='bold')
    ax8.axis('off')
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04, label='Depth diff (cm)')
    
    # Row 3: Error maps
    # 9. Sparse Error
    ax9 = plt.subplot(3, 4, 9)
    sparse_error_vis = np.ma.masked_where(~(sparse_mask & gt_mask), sparse_error)
    im9 = ax9.imshow(sparse_error_vis, cmap='hot', vmin=0, vmax=10)
    ax9.set_title(f'(i) Sparse Error\nMAE: {sparse_mae:.2f} cm', 
                  fontsize=12, fontweight='bold')
    ax9.axis('off')
    plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04, label='Error (cm)')
    
    # 10. Dense Error
    ax10 = plt.subplot(3, 4, 10)
    dense_error_vis = np.ma.masked_where(~gt_mask, dense_error)
    im10 = ax10.imshow(dense_error_vis, cmap='hot', vmin=0, vmax=10)
    ax10.set_title(f'(j) Dense Error\nMAE: {dense_mae:.2f} cm', 
                   fontsize=12, fontweight='bold')
    ax10.axis('off')
    plt.colorbar(im10, ax=ax10, fraction=0.046, pad=0.04, label='Error (cm)')
    
    # 11. Error Histogram
    ax11 = plt.subplot(3, 4, 11)
    if sparse_mask.sum() > 0 and gt_mask.sum() > 0:
        sparse_errors = sparse_error[sparse_mask & gt_mask]
        dense_errors = dense_error[gt_mask]
        ax11.hist(sparse_errors, bins=30, alpha=0.5, label='Sparse', color='blue', density=True)
        ax11.hist(dense_errors, bins=30, alpha=0.5, label='Dense', color='green', density=True)
        ax11.set_xlabel('Error (cm)', fontsize=10)
        ax11.set_ylabel('Density', fontsize=10)
        ax11.set_title('(k) Error Distribution', fontsize=12, fontweight='bold')
        ax11.legend()
        ax11.grid(True, alpha=0.3)
    
    # 12. Summary Statistics
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    # Create summary text
    summary_text = f"""
    IMAGE #{idx} SUMMARY
    {'='*30}
    
    COVERAGE:
      Sparse:  {sparse_coverage:5.1f}%
      Dense:   {dense_coverage:5.1f}%
      Gain:    +{dense_coverage - sparse_coverage:5.1f}%
    
    ACCURACY (RMSE):
      Sparse:  {sparse_rmse:5.2f} cm
      Dense:   {dense_rmse:5.2f} cm
      Change:  {((dense_rmse - sparse_rmse) / sparse_rmse * 100) if not np.isnan(sparse_rmse) else 0:+5.1f}%
    
    MEAN ABSOLUTE ERROR:
      Sparse:  {sparse_mae:5.2f} cm
      Dense:   {dense_mae:5.2f} cm
    
    KEY INSIGHT:
      {'✅ Dense improves!' if dense_rmse < sparse_rmse else '❌ Dense degrades slightly'}
      {'✅ Full coverage!' if dense_coverage == 100 else ''}
    """
    
    ax12.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
              verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Overall title
    fig.suptitle(f'Depth Densification Comparison - Test Image #{idx}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Depth Densifier Results')
    parser.add_argument('--data_path', type=str, default='./data_test/regular',
                        help='Path to test data')
    parser.add_argument('--model_path', type=str, default='./pretrained_weights/best_densifier.pth',
                        help='Path to trained densifier model')
    parser.add_argument('--start_idx', type=int, default=180,
                        help='Starting test image index')
    parser.add_argument('--num_images', type=int, default=5,
                        help='Number of images to visualize')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--cuda', type=str, default='cuda:0',
                        help='CUDA device')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load test data
    print("Loading test data...")
    images = np.load(os.path.join(args.data_path, 'images_ny.npy'))
    gt_depths = np.load(os.path.join(args.data_path, 'depth_maps.npy'))
    print(f"Loaded {len(images)} test images\n")
    
    # Load trained model
    print("Loading trained densifier model...")
    model = load_model(args.model_path, device)
    print("Model loaded!\n")
    
    print("="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Process images
    for i in range(args.start_idx, min(args.start_idx + args.num_images, len(images))):
        print(f"\nProcessing image #{i}...")
        
        # Load image and ground truth
        image = images[i, 0, :, :, :]
        gt_depth = gt_depths[i]
        
        # Normalize image
        if image.max() > 1.0:
            image = image / 255.0
        
        # Load sparse depth and confidence
        sparse_path = f'./logs/blurry_edges_depths/depth_{i:03d}.npy'
        conf_path = f'./logs/blurry_edges_depths/confidence_{i:03d}.npy'
        
        if not os.path.exists(sparse_path):
            print(f"  WARNING: Sparse depth not found, skipping...")
            continue
        
        sparse_depth = np.load(sparse_path)
        if len(sparse_depth.shape) == 3:
            sparse_depth = sparse_depth[0]
        
        if os.path.exists(conf_path):
            confidence = np.load(conf_path)
            if len(confidence.shape) == 3:
                confidence = confidence[0]
        else:
            confidence = np.ones_like(sparse_depth) * 0.5
        
        # Compute boundary
        boundary = compute_boundary(sparse_depth)
        
        # Prepare input for densifier
        input_tensor = np.stack([
            sparse_depth,
            boundary,
            confidence,
            image[:, :, 0],
            image[:, :, 1],
            image[:, :, 2]
        ], axis=0)[np.newaxis, :, :, :].astype(np.float32)
        
        input_tensor = torch.from_numpy(input_tensor).to(device)
        
        # Run densifier
        with torch.no_grad():
            dense_depth = model(input_tensor)
        
        dense_depth = dense_depth.cpu().numpy()[0, 0]
        
        # Create visualization
        save_path = os.path.join(args.output_dir, f'comparison_{i:03d}.png')
        visualize_comparison(i, image, sparse_depth, confidence, dense_depth, gt_depth, save_path)
    
    print("\n" + "="*80)
    print(f"✓ All visualizations saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
