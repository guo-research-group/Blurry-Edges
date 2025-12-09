"""
Modified version of blurry_edges_test.py that saves RAW depth predictions

This saves the global_depth_map BEFORE applying the confidence threshold,
so we can see what the baseline actually predicts everywhere.
"""

import numpy as np
import os
import cv2
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import TestDataset
from models import LocalStage, GlobalStage, DepthCompletion
from utils import get_args, DepthEtas, PostProcessGlobalBase, Visualizer, eval_depth

class PostProcess(PostProcessGlobalBase):
    def __init__(self, args, depthCal, device):
        super().__init__(args, device)
        self.depthCal = depthCal
        self.rho_prime = args.rho_prime
        self.densify = args.densify
    
    def get_colors(self, wedges, img_patches, colors_only):
        if colors_only:
            A = wedges.permute(0,4,5,2,3,1).reshape(self.batch_size * 2, self.H_patches, self.W_patches, -1, 3)
            y = img_patches.permute(0,4,5,2,3,1).reshape(self.batch_size * 2, self.H_patches, self.W_patches, -1, 3)
        else:
            A = wedges.permute(0,5,6,1,3,4,2).reshape(self.batch_size, self.H_patches, self.W_patches, -1, 3)
            y = img_patches.permute(0,5,6,1,3,4,2).reshape(self.batch_size, self.H_patches, self.W_patches, -1, 3)
        A_t = A.permute(0,1,2,4,3)
        colors = torch.matmul(self.inverse_3by3(torch.matmul(A_t, A)+self.ridge), torch.matmul(A_t, y)).permute(0,4,3,1,2)
        return colors

if __name__ == '__main__':
    args = get_args('eval')
    device = torch.device(args.cuda)
    
    print("="*80)
    print("GENERATING RAW BASELINE DEPTHS (Before Thresholding)")
    print("="*80)
    print(f"Using device: {device}")
    print(f"Data path: {args.data_path}")
    
    # Set default attributes if not present
    if not hasattr(args, 'depth_thres'):
        args.depth_thres = 0.05
    if not hasattr(args, 'local_path'):
        args.local_path = './pretrained_weights/pretrained_local_stage.pth'
    if not hasattr(args, 'global_path'):
        args.global_path = './pretrained_weights/pretrained_global_stage.pth'
    
    # Override data path to use regular test set
    args.data_path = './data_test/regular'
    
    print(f"Confidence threshold: {args.depth_thres}")
    
    # Load models
    print("\nLoading models...")
    local_module = LocalStage().to(device)
    global_module = GlobalStage().to(device)
    
    local_module.load_state_dict(torch.load(args.local_path, map_location=device))
    global_module.load_state_dict(torch.load(args.global_path, map_location=device))
    
    local_module.eval()
    global_module.eval()
    print("Models loaded successfully!")
    
    # Create helper
    depthCal = DepthEtas(args, device)
    helper = PostProcess(args, depthCal, device)
    helper.to(device)
    
    # Load test data  
    print("\nLoading test data...")
    test_dataset = TestDataset(device, data_path=args.data_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Loaded {len(test_dataset)} test images")
    
    # Output directory
    output_dir = './logs/blurry_edges_raw_actual'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process only test set images (180-199)
    start_idx = 180
    end_idx = 190  # Process 10 images for now
    
    print(f"\nProcessing images {start_idx} to {end_idx-1}...")
    print(f"Output directory: {output_dir}/")
    print("="*80 + "\n")
    
    depth_thres = args.depth_thres
    
    with torch.no_grad():
        for j, (img_pair, gt_depth) in enumerate(test_loader):
            if j < start_idx:
                continue
            if j >= end_idx:
                break
            
            # img_pair is already normalized by alpha
            img_pair = img_pair.to(device)
            gt_depth = gt_depth.to(device)
            
            # Create alpha_pair (ones since already normalized)
            alpha_pair = torch.ones(1, 2, *img_pair.shape[2:4]).to(device)
            
            # Extract patches
            vec, img_patches, colors = helper.image2vec(img_pair, alpha_pair)
            vec = vec.to(device)
            
            # LocalStage prediction
            params_est = local_module(vec)
            params_est = params_est.view(1, 2 * helper.H_patches * helper.W_patches, 10)
            
            # Prepare for GlobalStage
            xy = params_est[:, :, :4]
            angles = torch.remainder(params_est[:, :, 4:8], 2 * torch.pi)
            etas_coef = params_est[:, :, 8:]
            
            pm = torch.cat([xy / 3, \
                          (angles - torch.pi) / torch.pi, \
                          etas_coef - 0.5, \
                          (colors - 0.5) * 2], dim=2).unsqueeze(0).permute(0,2,1,3).flatten(2,3)
            
            # GlobalStage refinement
            params = global_module(pm)
            xy = params[:, :, :4] * 3
            angles = torch.remainder((params[:, :, 4:8] + 1) * torch.pi, 2 * torch.pi)
            etas_coef = params[:, :, 8:] + 0.5
            est = torch.cat([xy, angles, etas_coef], dim=2)
            
            # Post-processing - get depth and confidence
            col_est, col_shpd, col_refoc, bndry_est, global_depth_map, confidence_map = helper(
                est, img_patches, colors_only=False
            )
            
            # ===== KEY: Save RAW depth (before thresholding) =====
            raw_depth = global_depth_map  # This is what baseline ACTUALLY predicts!
            
            # Also compute thresholded version
            thresholded_depth = np.where(confidence_map > depth_thres, 
                                        global_depth_map, 
                                        np.zeros_like(global_depth_map))
            
            # Save both versions
            np.save(f'{output_dir}/raw_depth_{j:03d}.npy', raw_depth)
            np.save(f'{output_dir}/thresholded_depth_{j:03d}.npy', thresholded_depth)
            np.save(f'{output_dir}/confidence_{j:03d}.npy', confidence_map)
            
            # Statistics
            raw_coverage = (raw_depth > 0).sum() / raw_depth.size * 100
            thresh_coverage = (thresholded_depth > 0).sum() / thresholded_depth.size * 100
            low_conf_pixels = (confidence_map <= depth_thres).sum() / confidence_map.size * 100
            
            print(f"Image {j:03d}:")
            print(f"  Raw depth: min={raw_depth.min():.3f}, max={raw_depth.max():.3f}, mean={raw_depth.mean():.3f}")
            print(f"  Confidence: min={confidence_map.min():.3f}, max={confidence_map.max():.3f}, mean={confidence_map.mean():.3f}")
            print(f"  Raw coverage: {raw_coverage:.1f}%")
            print(f"  Thresholded coverage: {thresh_coverage:.1f}%")
            print(f"  Low-conf pixels (discarded): {low_conf_pixels:.1f}%")
            print()
    
    print("="*80)
    print("DONE!")
    print("="*80)
    print(f"Saved {end_idx - start_idx} images to: {output_dir}/")
    print("\nFiles per image:")
    print("  - raw_depth_XXX.npy: RAW baseline predictions (no threshold)")
    print("  - thresholded_depth_XXX.npy: Thresholded predictions (conf > 0.05)")
    print("  - confidence_XXX.npy: Confidence maps")
    print("\nNow run compare_raw_vs_densifier.py to see the difference!")
    print("="*80)
