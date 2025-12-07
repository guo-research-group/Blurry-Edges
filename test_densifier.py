"""
Testing Script for Depth Densifier
====================================
Evaluates the trained densifier on test set and compares:
1. Blurry-Edges sparse depth (baseline)
2. Densified depth (our contribution)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models.depth_densifier import DepthDensifierUNet
from utils.metrics import compute_errors


def load_model(checkpoint_path, device):
    """Load trained densifier model"""
    model = DepthDensifierUNet(in_channels=6, out_channels=1)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    if 'rmse' in checkpoint:
        print(f"  Training RMSE: {checkpoint['rmse']:.4f}")
    
    return model


def compute_boundary(depth):
    """Compute boundary map from depth using Sobel filter"""
    from scipy.ndimage import sobel
    dx = sobel(depth, axis=0)
    dy = sobel(depth, axis=1)
    boundary = np.sqrt(dx**2 + dy**2)
    boundary = (boundary - boundary.min()) / (boundary.max() - boundary.min() + 1e-8)
    return boundary


def main():
    parser = argparse.ArgumentParser(description='Test Depth Densifier')
    parser.add_argument('--data_path', type=str, default='./data_test/regular',
                        help='Path to test data')
    parser.add_argument('--model_path', type=str, default='./pretrained_weights/best_densifier.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--num_images', type=int, default=20,
                        help='Number of test images (default: 20 for held-out test set)')
    parser.add_argument('--start_idx', type=int, default=180,
                        help='Starting index for test images (default: 180 for held-out test set)')
    parser.add_argument('--cuda', type=str, default='cuda:0',
                        help='CUDA device')
    parser.add_argument('--save_outputs', action='store_true',
                        help='Save densified depth maps')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load test data
    print("Loading test data...")
    images = np.load(os.path.join(args.data_path, 'images_ny.npy'))
    # Load actual depth maps, not scalar alphas
    gt_depths = np.load(os.path.join(args.data_path, 'depth_maps.npy'))
    
    print(f"Loaded {len(images)} test images")
    
    # Load trained model
    print("\nLoading trained densifier model...")
    model = load_model(args.model_path, device)
    
    # Create output directory if saving
    if args.save_outputs:
        output_dir = './logs/densified_depths'
        os.makedirs(output_dir, exist_ok=True)
    
    # Metrics storage
    sparse_metrics = {'delta1': [], 'rmse': [], 'abs_rel': []}
    dense_metrics = {'delta1': [], 'rmse': [], 'abs_rel': []}
    
    print("\n" + "="*80)
    print("DEPTH DENSIFICATION EVALUATION (HELD-OUT TEST SET)")
    print("="*80)
    print(f"Testing on indices {args.start_idx} to {args.start_idx + args.num_images - 1}")
    print(f"These images were NOT used during training or validation!")
    print("="*80 + "\n")
    
    # Process each image from held-out test set
    test_indices = range(args.start_idx, min(args.start_idx + args.num_images, len(images)))
    
    for i in tqdm(test_indices, desc="Processing images"):
        print(f"\n{'='*80}")
        print(f"Image #{i} (Test set - never seen during training)")
        print(f"{'='*80}")
        
        # Load image and ground truth
        image = images[i, 0, :, :, :]  # Use first view
        gt_depth = gt_depths[i]
        
        # Normalize image
        if image.max() > 1.0:
            image = image / 255.0
        
        # Load sparse Blurry-Edges depth
        sparse_path = f'./logs/blurry_edges_depths/depth_{i:03d}.npy'
        conf_path = f'./logs/blurry_edges_depths/confidence_{i:03d}.npy'
        
        if not os.path.exists(sparse_path):
            print(f"  WARNING: Sparse depth not found at {sparse_path}")
            print(f"  Run: python blurry_edges_test.py first")
            continue
        
        sparse_depth = np.load(sparse_path)
        if len(sparse_depth.shape) == 3:
            sparse_depth = sparse_depth[0]
        
        # Load or create confidence map
        if os.path.exists(conf_path):
            confidence = np.load(conf_path)
            if len(confidence.shape) == 3:
                confidence = confidence[0]
        else:
            confidence = np.ones_like(sparse_depth) * 0.5
        
        # Compute boundary map
        boundary = compute_boundary(sparse_depth)
        
        # Prepare input tensor [1, 6, H, W]
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
        
        dense_depth = dense_depth.cpu().numpy()[0, 0]  # [H, W]
        
        # Create valid mask
        valid_mask = gt_depth > 0
        
        if valid_mask.sum() == 0:
            print(f"  WARNING: No valid ground truth pixels")
            continue
        
        # Compute metrics for sparse depth
        sparse_valid = (sparse_depth > 0) & valid_mask
        if sparse_valid.sum() > 10:
            sparse_result = compute_errors(gt_depth[sparse_valid], sparse_depth[sparse_valid])
            sparse_metrics['delta1'].append(sparse_result['delta1'])
            sparse_metrics['rmse'].append(sparse_result['rmse'])
            sparse_metrics['abs_rel'].append(sparse_result['abs_rel'])
            
            print(f"Sparse (Blurry-Edges):")
            print(f"  Coverage: {sparse_valid.sum() / valid_mask.sum() * 100:.1f}% of pixels")
            print(f"  RMSE:     {sparse_result['rmse']:.4f}")
            print(f"  delta1:   {sparse_result['delta1']:.4f}")
            print(f"  AbsRel:   {sparse_result['abs_rel']:.4f}")
        else:
            print(f"  Sparse: Insufficient valid pixels")
        
        # Compute metrics for dense depth
        dense_result = compute_errors(gt_depth[valid_mask], dense_depth[valid_mask])
        dense_metrics['delta1'].append(dense_result['delta1'])
        dense_metrics['rmse'].append(dense_result['rmse'])
        dense_metrics['abs_rel'].append(dense_result['abs_rel'])
        
        print(f"\nDense (Densified):")
        print(f"  Coverage: 100.0% of pixels")
        print(f"  RMSE:     {dense_result['rmse']:.4f}")
        print(f"  delta1:   {dense_result['delta1']:.4f}")
        print(f"  AbsRel:   {dense_result['abs_rel']:.4f}")
        
        # Compute improvement
        if sparse_valid.sum() > 10:
            rmse_improvement = (sparse_result['rmse'] - dense_result['rmse']) / sparse_result['rmse'] * 100
            delta1_improvement = (dense_result['delta1'] - sparse_result['delta1']) / sparse_result['delta1'] * 100
            
            print(f"\nImprovement:")
            print(f"  RMSE:   {rmse_improvement:+.1f}%")
            print(f"  delta1: {delta1_improvement:+.1f}%")
        
        # Save densified depth
        if args.save_outputs:
            np.save(f'{output_dir}/dense_{i:03d}.npy', dense_depth)
    
    # Print final results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    if len(sparse_metrics['rmse']) > 0:
        print(f"\nSparse (Blurry-Edges only):")
        print(f"  RMSE:   {np.mean(sparse_metrics['rmse']):.4f} ± {np.std(sparse_metrics['rmse']):.4f}")
        print(f"  delta1: {np.mean(sparse_metrics['delta1']):.4f} ± {np.std(sparse_metrics['delta1']):.4f}")
        print(f"  AbsRel: {np.mean(sparse_metrics['abs_rel']):.4f} ± {np.std(sparse_metrics['abs_rel']):.4f}")
    
    if len(dense_metrics['rmse']) > 0:
        print(f"\nDense (with Densifier):")
        print(f"  RMSE:   {np.mean(dense_metrics['rmse']):.4f} ± {np.std(dense_metrics['rmse']):.4f}")
        print(f"  delta1: {np.mean(dense_metrics['delta1']):.4f} ± {np.std(dense_metrics['delta1']):.4f}")
        print(f"  AbsRel: {np.mean(dense_metrics['abs_rel']):.4f} ± {np.std(dense_metrics['abs_rel']):.4f}")
    
    if len(sparse_metrics['rmse']) > 0 and len(dense_metrics['rmse']) > 0:
        rmse_improve = (np.mean(sparse_metrics['rmse']) - np.mean(dense_metrics['rmse'])) / np.mean(sparse_metrics['rmse']) * 100
        delta1_improve = (np.mean(dense_metrics['delta1']) - np.mean(sparse_metrics['delta1'])) / np.mean(sparse_metrics['delta1']) * 100
        
        print(f"\nOverall Improvement:")
        print(f"  RMSE:   {rmse_improve:+.2f}%")
        print(f"  delta1: {delta1_improve:+.2f}%")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
