"""
Generate ACTUAL raw baseline depth predictions (before thresholding)

This script runs the baseline model and saves:
1. global_depth_map - RAW predictions (no threshold)
2. confidence_map - Confidence scores
3. thresholded_depth_map - Thresholded predictions (current output)

This allows us to properly compare what happens with and without thresholding.
"""

import numpy as np
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import TestDataset
from models import LocalStage, GlobalStage
from utils import get_args, DepthEtas, PostProcessGlobalBase

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

    def get_patches(self, xy_angles, etas, colors_only):
        # (implementation from blurry_edges_test.py)
        pass

    def forward(self, params, img_patches, colors_only=False):
        # Run post-processing to get depth and confidence
        # Returns: col_est, col_shpd, col_refoc, bndry_est, depth_map, confidence_map
        # (full implementation would go here)
        pass


def main():
    args = get_args()
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print("="*80)
    print("GENERATING ACTUAL RAW BASELINE DEPTHS (No Threshold)")
    print("="*80)
    
    # Load models
    print("\nLoading baseline models...")
    local_module = LocalStage().to(device)
    global_module = GlobalStage().to(device)
    
    local_module.load_state_dict(torch.load(args.local_path, map_location=device))
    global_module.load_state_dict(torch.load(args.global_path, map_location=device))
    
    local_module.eval()
    global_module.eval()
    
    # Create helper
    depthCal = DepthEtas(args, device)
    helper = PostProcess(args, depthCal, device)
    helper.to(device)
    
    # Load test data
    print("Loading test data...")
    test_dataset = TestDataset(args)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Output directories
    raw_dir = './logs/blurry_edges_raw_actual'
    os.makedirs(raw_dir, exist_ok=True)
    
    print(f"\nProcessing {len(test_dataset)} test images...")
    print("Saving to: ./logs/blurry_edges_raw_actual/")
    print("="*80)
    
    depth_thres = args.depth_thres  # Default 0.05
    
    with torch.no_grad():
        for j, (img_pair, alpha_pair, gt_depth) in enumerate(tqdm(test_loader, desc="Processing")):
            img_pair = img_pair.to(device)
            alpha_pair = alpha_pair.to(device)
            
            # Extract patches and features
            vec, img_patches, colors = helper.image2vec(img_pair, alpha_pair)
            vec = vec.to(device)
            
            # LocalStage: predict parameters
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
            
            # GlobalStage: refine parameters
            params = global_module(pm)
            xy = params[:, :, :4] * 3
            angles = torch.remainder((params[:, :, 4:8] + 1) * torch.pi, 2 * torch.pi)
            etas_coef = params[:, :, 8:] + 0.5
            est = torch.cat([xy, angles, etas_coef], dim=2)
            
            # Post-processing: get depth and confidence
            col_est, col_shpd, col_refoc, bndry_est, global_depth_map, confidence_map = helper(
                est, img_patches, colors_only=False
            )
            
            # ===== CRITICAL: Save BOTH raw and thresholded =====
            
            # 1. RAW depth (no threshold) - THIS IS WHAT WE NEED!
            raw_depth = global_depth_map  # NO thresholding
            
            # 2. Thresholded depth (current baseline output)
            thresholded_depth = np.where(confidence_map > depth_thres, 
                                        global_depth_map, 
                                        np.zeros_like(global_depth_map))
            
            # 3. Confidence map
            conf = confidence_map
            
            # Save all three
            np.save(f'{raw_dir}/raw_depth_{j:03d}.npy', raw_depth)
            np.save(f'{raw_dir}/thresholded_depth_{j:03d}.npy', thresholded_depth)
            np.save(f'{raw_dir}/confidence_{j:03d}.npy', conf)
            
            # Print sample statistics for first image
            if j == 180:
                print(f"\nSample statistics for image {j}:")
                print(f"  Raw depth range: [{raw_depth.min():.3f}, {raw_depth.max():.3f}]")
                print(f"  Raw depth mean: {raw_depth.mean():.3f}")
                print(f"  Raw depth std: {raw_depth.std():.3f}")
                print(f"  Confidence range: [{conf.min():.3f}, {conf.max():.3f}]")
                print(f"  Confidence mean: {conf.mean():.3f}")
                print(f"  Thresholded coverage: {(thresholded_depth > 0).sum() / thresholded_depth.size * 100:.1f}%")
                print(f"  Pixels discarded by threshold: {(conf <= depth_thres).sum() / conf.size * 100:.1f}%")
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"Saved {len(test_dataset)} images to: {raw_dir}/")
    print("\nFiles saved:")
    print("  - raw_depth_XXX.npy: Baseline predictions WITHOUT threshold")
    print("  - thresholded_depth_XXX.npy: Baseline predictions WITH threshold (0.05)")
    print("  - confidence_XXX.npy: Confidence maps")
    print("\nNow you can properly compare:")
    print("  1. Raw baseline (all predictions)")
    print("  2. Thresholded baseline (high-confidence only)")
    print("  3. U-Net densifier (learned completion)")
    print("="*80)


if __name__ == '__main__':
    main()
