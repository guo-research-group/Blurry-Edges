"""
Simple Depth Fusion Evaluation
Combines Blurry-Edges depth + MiDaS monocular depth
"""
import os
import sys
import torch
import numpy as np
import argparse
import time
from torch.utils.data import DataLoader
import torch.nn as nn

from data import TestDataset
from models import LocalStage, GlobalStage
from utils import get_args, DepthEtas, PostProcessGlobalBase
from depth_fusion import fuse_single_image
from utils.metrics import compute_errors


def align_midas_scale(midas_depth, gt_depth):
    """Align MiDaS to ground truth scale"""
    midas_inv = 1.0 / (midas_depth + 1e-6)
    mask = gt_depth > 0
    if mask.sum() > 100:
        scale = np.median(gt_depth[mask] / (midas_inv[mask] + 1e-6))
        aligned = midas_inv * scale
        shift = np.median(gt_depth[mask] - aligned[mask])
        return aligned + shift
    return midas_inv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data_test/regular')
    parser.add_argument('--midas_path', type=str, default='./midas_predictions')
    parser.add_argument('--cuda', type=str, default='cuda:0')
    parser.add_argument('--num_images', type=int, default=10)
    parser.add_argument('--lambda1', type=float, default=10.0)
    parser.add_argument('--lambda2', type=float, default=1.0)
    parser.add_argument('--lambda3', type=float, default=0.1)
    args_fusion = parser.parse_args()
    
    device = torch.device(args_fusion.cuda if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load datasets
    test_dataset = TestDataset(device=device, data_path=args_fusion.data_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Loaded {len(test_dataset)} image pairs\n")
    
    # Load MiDaS
    midas_file = os.path.join(args_fusion.midas_path, 'midas_depths.npy')
    if not os.path.exists(midas_file):
        raise FileNotFoundError(f"Run: python run_midas.py first!")
    midas_depths = np.load(midas_file)[:args_fusion.num_images]
    print(f"Loaded MiDaS: {midas_depths.shape}\n")
    
    # Load Blurry-Edges models
    sys.argv = ['test_fusion.py', '--cuda', args_fusion.cuda]
    args = get_args('eval', big=False)
    
    local_model = LocalStage().to(device)
    global_model = GlobalStage(in_parameter_size=38, out_parameter_size=12, device=device).to(device)
    local_model.load_state_dict(torch.load(f'{args.model_path}/pretrained_local_stage.pth', map_location=device))
    global_model.load_state_dict(torch.load(f'{args.model_path}/pretrained_global_stage.pth', map_location=device))
    local_model.eval()
    global_model.eval()
    
    depthCal = DepthEtas(args, device)
    
    # Simplified PostProcess
    class SimplePost(PostProcessGlobalBase):
        def __init__(self, args, depthCal, device):
            super().__init__(args, device)
            self.depthCal = depthCal
        def forward(self, params, img_patches):
            params = params.permute(0,2,1).view(1, 12, self.H_patches, self.W_patches)
            dists = self.depthCal.params2dists(params.permute(0,2,3,1))
            depth_map, depth_mask = self.depthCal.dists2depths(dists)
            global_depth, confidence = self.local2global_depth(depth_map, depth_mask)
            return global_depth.cpu().numpy(), confidence.cpu().numpy()
    
    post_proc = SimplePost(args, depthCal, device)
    print("Models loaded!\n")
    
    print(f"{'='*80}\nDEPTH FUSION EVALUATION\n{'='*80}\n")
    
    results_be, results_midas, results_fused = [], [], []
    
    for i, (img_ny, depth_gt) in enumerate(test_loader):
        if i >= args_fusion.num_images:
            break
            
        print(f"{'='*80}\nImage #{i}\n{'='*80}")
        start = time.time()
        
        img_ny_np = img_ny.cpu().numpy()[0]
        depth_gt_np = depth_gt.cpu().numpy()[0]
        img_rgb = img_ny_np[0]
        
        # Blurry-Edges inference
        t_img = img_ny.flatten(0,1).permute(0,3,1,2)
        H_patches = (args.img_size[0] - args.R) // args.stride + 1
        W_patches = H_patches
        img_patches = nn.Unfold(args.R, stride=args.stride)(t_img).view(2, 3, args.R, args.R, H_patches, W_patches)
        
        # Reshape for local stage: [2*H*W, 3, R, R]
        vec = img_patches.permute(0,4,5,1,2,3).reshape(2 * H_patches * W_patches, 3, args.R, args.R)
        params_est = local_model(vec.to(torch.float32))
        
        # Reshape back and prepare for global stage
        local_params = params_est.view(2, H_patches, W_patches, 10).flatten(start_dim=1,end_dim=2).detach()
        global_params = global_model(local_params[:,:,:-1], H_patches, W_patches)
        
        Z_BE, confidence = post_proc(global_params, img_patches)
        Z_BE_filtered = np.where(confidence > 0.05, Z_BE, 0)
        
        # MiDaS
        Z_midas = midas_depths[i]
        
        print(f"Blurry-Edges: [{Z_BE_filtered.min():.2f}, {Z_BE_filtered.max():.2f}]")
        print(f"MiDaS:        [{Z_midas.min():.2f}, {Z_midas.max():.2f}]")
        
        # Fusion
        Z_fused = fuse_single_image(Z_BE_filtered, Z_midas, confidence, img_rgb,
                                     lambda1=args_fusion.lambda1, lambda2=args_fusion.lambda2,
                                     lambda3=args_fusion.lambda3, device=device, verbose=False)
        
        # Metrics
        Z_midas_aligned = align_midas_scale(Z_midas, depth_gt_np)
        valid = depth_gt_np > 0
        
        err_be = compute_errors(depth_gt_np[valid], Z_BE_filtered[valid])
        err_midas = compute_errors(depth_gt_np[valid], Z_midas_aligned[valid])
        err_fused = compute_errors(depth_gt_np[valid], Z_fused[valid])
        
        results_be.append(err_be)
        results_midas.append(err_midas)
        results_fused.append(err_fused)
        
        print(f"\n{'Method':<15} {'delta1':>8} {'RMSE':>8} {'AbsRel':>8}")
        print(f"{'-'*47}")
        print(f"{'Blurry-Edges':<15} {err_be['delta1']:>8.3f} {err_be['rmse']:>8.2f} {err_be['abs_rel']:>8.2f}")
        print(f"{'MiDaS':<15} {err_midas['delta1']:>8.3f} {err_midas['rmse']:>8.2f} {err_midas['abs_rel']:>8.2f}")
        print(f"{'Fused':<15} {err_fused['delta1']:>8.3f} {err_fused['rmse']:>8.2f} {err_fused['abs_rel']:>8.2f}")
        
        improve = (err_be['rmse'] - err_fused['rmse']) / err_be['rmse'] * 100
        print(f"\nImprovement: {improve:+.1f}% | Time: {time.time()-start:.2f}s\n")
    
    # Final results
    avg_be = {k: np.mean([r[k] for r in results_be]) for k in results_be[0].keys()}
    avg_midas = {k: np.mean([r[k] for r in results_midas]) for k in results_midas[0].keys()}
    avg_fused = {k: np.mean([r[k] for r in results_fused]) for k in results_fused[0].keys()}
    
    print(f"\n{'='*80}\nFINAL RESULTS\n{'='*80}\n")
    print(f"{'Method':<15} {'delta1':>8} {'RMSE':>8} {'AbsRel':>8}")
    print(f"{'-'*47}")
    print(f"{'Blurry-Edges':<15} {avg_be['delta1']:>8.3f} {avg_be['rmse']:>8.2f} {avg_be['abs_rel']:>8.2f}")
    print(f"{'MiDaS':<15} {avg_midas['delta1']:>8.3f} {avg_midas['rmse']:>8.2f} {avg_midas['abs_rel']:>8.2f}")
    print(f"{'Fused':<15} {avg_fused['delta1']:>8.3f} {avg_fused['rmse']:>8.2f} {avg_fused['abs_rel']:>8.2f}")
    
    improve = (avg_be['rmse'] - avg_fused['rmse']) / avg_be['rmse'] * 100
    print(f"\n{'='*80}")
    print(f"OVERALL IMPROVEMENT: {improve:+.1f}%")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
