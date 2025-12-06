"""
Test Depth Fusion: Evaluate Blurry-Edges + MiDaS fusion
Compares three methods:
1. Blurry-Edges only (baseline)
2. MiDaS only 
3. Fused (our contribution)
"""
import os
import torch
import numpy as np
import argparse
import time
from data import TestDataset
from depth_fusion import fuse_single_image
from utils.metrics import compute_errors
from utils import get_args
import torch.nn as nn

# Import helper class from blurry_edges_test
import sys
sys.path.insert(0, '.')
from blurry_edges_test import Helper

class DepthEstimator:
    """Simplified depth estimator using existing Helper class"""
    def __init__(self, device):
        # Use get_args to get proper configuration
        sys.argv = ['test_fusion.py', '--cuda', str(device), '--data_path', './data_test/regular']
        args = get_args('eval', big=False)
        args.cuda = str(device)
        args.R = 27
        args.stride = 9
        
        # Create helper (this handles all model loading and inference)
        self.helper = Helper(args, device)
        self.device = device
        
    def get_patches(self, img_ny, img_size):
        """Extract patches from image"""
        t_img = img_ny.flatten(0, 1).permute(0, 3, 1, 2)
        H_patches = (img_size - self.R) // self.stride + 1
        W_patches = H_patches
        img_patches = nn.Unfold(self.R, stride=self.stride)(t_img).view(
            2, 3, self.R, self.R, H_patches, W_patches
        )
        return img_patches
    
    def helper(self, est, img_patches, colors_only=False):
        """Run depth estimation"""
    def get_patches(self, img_ny, img_size):
        """Extract patches using helper"""
        return self.helper.get_patches(img_ny, img_size)
def align_midas_scale(midas_depth, gt_depth):
    """
    Align MiDaS depth to ground truth scale for fair comparison
    
    Args:
        midas_depth: MiDaS depth map
        gt_depth: Ground truth depth map
        
    Returns:
        Aligned MiDaS depth
    """
    # MiDaS outputs inverse depth
    # Load Blurry-Edges models
    print("\nLoading Blurry-Edges models...")
    est = DepthEstimator(device=device)
    print("Models loaded!")
    if mask.sum() > 100:
        scale = np.median(gt_depth[mask] / (midas_depth_inv[mask] + 1e-6))
        midas_aligned = midas_depth_inv * scale
        
        # Also align shift
        shift = np.median(gt_depth[mask] - midas_aligned[mask])
        midas_aligned = midas_aligned + shift
    else:
        # Fallback
        midas_aligned = midas_depth_inv
        
    return midas_aligned

def main():
    parser = argparse.ArgumentParser(description='Test Depth Fusion')
    parser.add_argument('--data_path', type=str, default='./data_test/regular',
                        help='Path to test data')
    parser.add_argument('--midas_path', type=str, default='./midas_predictions',
                        help='Path to MiDaS predictions')
    parser.add_argument('--cuda', type=str, default='cuda:0',
                        help='CUDA device')
    parser.add_argument('--num_images', type=int, default=10,
                        help='Number of images to test')
    parser.add_argument('--lambda1', type=float, default=10.0,
                        help='Weight for boundary term (Blurry-Edges)')
    parser.add_argument('--lambda2', type=float, default=1.0,
                        help='Weight for data term (monocular)')
    parser.add_argument('--lambda3', type=float, default=0.1,
                        help='Weight for smoothness term')
    parser.add_argument('--num_iterations', type=int, default=50,
                        help='Number of fusion iterations')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test dataset
    print(f"\nLoading test dataset from {args.data_path}")
    test_dataset = TestDataset(device=device, data_path=args.data_path)
    print(f"Loaded {len(test_dataset)} image pairs")
    
    # Load MiDaS predictions
    print(f"\nLoading MiDaS predictions from {args.midas_path}")
    midas_depths = load_midas_predictions(args.midas_path, args.num_images)
    
    # Load Blurry-Edges models
    print("\nLoading Blurry-Edges models...")
    est = DepthEstimator(device=device)
    print("Models loaded!")
    
    print(f"\n{'='*80}")
    print(f"DEPTH FUSION EVALUATION")
    print(f"{'='*80}")
    print(f"Fusion parameters: λ1={args.lambda1}, λ2={args.lambda2}, λ3={args.lambda3}")
    print(f"Iterations: {args.num_iterations}")
    print(f"{'='*80}\n")
    
    # Storage for results
    results_be = []      # Blurry-Edges only
    results_midas = []   # MiDaS only
    results_fused = []   # Fused
    
    for i in range(min(args.num_images, len(test_dataset))):
        print(f"\n{'='*80}")
        print(f"Image pair #{i}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Get data
        img_ny, depth_gt = test_dataset[i]
        img_ny_np = img_ny.cpu().numpy()
        depth_gt_np = depth_gt.cpu().numpy()
        
        # Get image for edge-aware smoothing
        img_rgb = img_ny_np[0]  # Use first defocused image
        
        # Run Blurry-Edges
        print("Running Blurry-Edges...")
        img_patches = est.get_patches(img_ny, img_ny.shape[2])
        _, _, _, _, global_depth_map, confidence_map = est.helper(
            est, img_patches, colors_only=False
        )
        
        # Convert to numpy
        Z_BE = global_depth_map
        confidence = confidence_map
        
        # Filter by confidence
        Z_BE_filtered = np.where(confidence > 0.05, Z_BE, 0)
        
        # Get MiDaS depth
        Z_midas = midas_depths[i]
        
        print(f"  Blurry-Edges depth range: [{Z_BE_filtered.min():.2f}, {Z_BE_filtered.max():.2f}]")
        print(f"  MiDaS depth range: [{Z_midas.min():.2f}, {Z_midas.max():.2f}]")
        print(f"  Ground truth range: [{depth_gt_np.min():.2f}, {depth_gt_np.max():.2f}]")
        
        # Fuse depths
        print("Fusing depths...")
        Z_fused = fuse_single_image(
            Z_BE_filtered, Z_midas, confidence, img_rgb,
        # Run Blurry-Edges
        print("Running Blurry-Edges...")
        img_patches = est.get_patches(img_ny, img_ny.shape[2])
        _, _, _, _, global_depth_map, confidence_map = est.helper(
            img_patches, colors_only=False
        )_midas_aligned = align_midas_scale(Z_midas, depth_gt_np)
        
        # Compute metrics for all three methods
        # Only evaluate where ground truth is valid
        valid_mask = depth_gt_np > 0
        
        # Blurry-Edges
        errors_be = compute_errors(depth_gt_np[valid_mask], Z_BE_filtered[valid_mask])
        results_be.append(errors_be)
        
        # MiDaS
        errors_midas = compute_errors(depth_gt_np[valid_mask], Z_midas_aligned[valid_mask])
        results_midas.append(errors_midas)
        
        # Fused
        errors_fused = compute_errors(depth_gt_np[valid_mask], Z_fused[valid_mask])
        results_fused.append(errors_fused)
        
        elapsed = time.time() - start_time
        
        # Print results
        print(f"\n{'='*80}")
        print(f"Results for Image #{i}:")
        print(f"{'='*80}")
        print(f"{'Method':<20} {'delta1':>10} {'RMSE (cm)':>12} {'AbsRel (cm)':>12}")
        print(f"{'-'*80}")
        print(f"{'Blurry-Edges':<20} {errors_be['delta1']:>10.3f} {errors_be['rmse']:>12.2f} {errors_be['abs_rel']:>12.2f}")
        print(f"{'MiDaS':<20} {errors_midas['delta1']:>10.3f} {errors_midas['rmse']:>12.2f} {errors_midas['abs_rel']:>12.2f}")
        print(f"{'Fused (Ours)':<20} {errors_fused['delta1']:>10.3f} {errors_fused['rmse']:>12.2f} {errors_fused['abs_rel']:>12.2f}")
        print(f"{'-'*80}")
        
        # Compute improvement
        rmse_improve = (errors_be['rmse'] - errors_fused['rmse']) / errors_be['rmse'] * 100
        delta1_improve = (errors_fused['delta1'] - errors_be['delta1']) / errors_be['delta1'] * 100
        
        print(f"Improvement over Blurry-Edges: RMSE {rmse_improve:+.1f}%, delta1 {delta1_improve:+.1f}%")
        print(f"Processing time: {elapsed:.2f}s")
    
    # Compute average metrics
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS (averaged over {len(results_be)} images)")
    print(f"{'='*80}")
    
    def average_results(results):
        avg = {}
        for key in results[0].keys():
            avg[key] = np.mean([r[key] for r in results])
        return avg
    
    avg_be = average_results(results_be)
    avg_midas = average_results(results_midas)
    avg_fused = average_results(results_fused)
    
    print(f"\n{'Method':<20} {'delta1':>10} {'RMSE (cm)':>12} {'AbsRel (cm)':>12}")
    print(f"{'-'*80}")
    print(f"{'Blurry-Edges':<20} {avg_be['delta1']:>10.3f} {avg_be['rmse']:>12.2f} {avg_be['abs_rel']:>12.2f}")
    print(f"{'MiDaS':<20} {avg_midas['delta1']:>10.3f} {avg_midas['rmse']:>12.2f} {avg_midas['abs_rel']:>12.2f}")
    print(f"{'Fused (Ours)':<20} {avg_fused['delta1']:>10.3f} {avg_fused['rmse']:>12.2f} {avg_fused['abs_rel']:>12.2f}")
    print(f"{'-'*80}")
    
    # Overall improvement
    rmse_improve = (avg_be['rmse'] - avg_fused['rmse']) / avg_be['rmse'] * 100
    delta1_improve = (avg_fused['delta1'] - avg_be['delta1']) / avg_be['delta1'] * 100
    
    print(f"\n{'='*80}")
    print(f"OVERALL IMPROVEMENT (Fused vs Blurry-Edges):")
    print(f"{'='*80}")
    print(f"RMSE:   {avg_be['rmse']:.2f} cm → {avg_fused['rmse']:.2f} cm ({rmse_improve:+.1f}%)")
    print(f"delta1: {avg_be['delta1']:.3f} → {avg_fused['delta1']:.3f} ({delta1_improve:+.1f}%)")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
