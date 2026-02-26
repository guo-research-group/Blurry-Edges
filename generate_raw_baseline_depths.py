"""
Generate RAW Blurry-Edges depth (no thresholding) for fair 100% coverage comparison.

This script saves:
1. global_depth_raw_{idx}.npy - Raw depth everywhere (100% coverage, includes noisy regions)
2. confidence_{idx}.npy - Confidence map (for reference)

This allows us to compare:
- Threshold=0.0 on RAW depth (100% coverage, noisy)
- Neural densifier (100% coverage, learned)
"""

import numpy as np
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import LocalStage, GlobalStage
from utils import get_args, DepthEtas, PostProcessGlobalBase


class TestDataset(torch.utils.data.Dataset):
    """Simple test dataset loader"""
    def __init__(self, device, data_path='./data_test/regular'):
        self.device = device
        self.images = np.load(os.path.join(data_path, 'images_ny.npy'))
        self.depths = np.load(os.path.join(data_path, 'depth_maps.npy'))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        depth = self.depths[idx]
        
        # Normalize if needed
        alphas_path = os.path.join(os.path.dirname(os.path.join('./data_test/regular', 'images_ny.npy')), 'alphas.npy')
        if os.path.exists(alphas_path):
            alphas = np.load(alphas_path)
            if len(alphas.shape) == 1:  # Scalar alphas
                alpha = alphas[idx]
                if alpha > 0:
                    image = image / alpha
        
        return torch.from_numpy(image).float(), torch.from_numpy(depth).float()

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
        dists = self.params2dists(xy_angles)
        if colors_only:
            wedges = self.dists2indicators(dists, etas)
            colors = self.get_colors(wedges, self.img_patches, colors_only)
        else:
            wedges1 = self.dists2indicators(dists, etas[:,:2,:,:])
            wedges2 = self.dists2indicators(dists, etas[:,2:,:,:])
            wedges = torch.cat([wedges1.unsqueeze(1), wedges2.unsqueeze(1)], dim=1)
            colors = self.get_colors(wedges, self.img_patches, colors_only)
            patches1 = (wedges1.unsqueeze(1) * colors.unsqueeze(-3).unsqueeze(-3)).sum(dim=2)
            patches2 = (wedges2.unsqueeze(1) * colors.unsqueeze(-3).unsqueeze(-3)).sum(dim=2)
            patches = torch.cat([patches1.unsqueeze(1), patches2.unsqueeze(1)], dim=1)

            depth_1 = self.depthCal.etas2depth(etas[:,0,:,:], etas[:,2,:,:])
            depth_2 = self.depthCal.etas2depth(etas[:,1,:,:], etas[:,3,:,:])
        
            if self.densify == 'w':
                depth_mask = (dists[:,0,...] > 0).to(torch.int32)
                depth_mask_temp = (dists[:,1,...] > 0).to(torch.int32) * 2
                depth_mask = torch.where(depth_mask_temp == 2, depth_mask_temp, depth_mask)
            else:
                depth_mask = (self.normalized_gaussian(dists[:,0,...]) > 0.5).to(torch.int32)
                depth_mask_temp = (self.normalized_gaussian(dists[:,1,...]) > 0.5).to(torch.int32) * 2
                depth_mask = torch.where((depth_mask_temp == 2) | (dists[:,1,...] >= 0), depth_mask_temp, depth_mask)
            
            depth_map = torch.where(depth_mask == 1, depth_1.unsqueeze(1).unsqueeze(1), \
                                    torch.where(depth_mask == 2, depth_2.unsqueeze(1).unsqueeze(1), depth_mask))

            dists_B = torch.where(dists[:,1,...] >= 0, dists[:,1,...], \
                                  torch.where(torch.abs(dists[:,0,...])<torch.abs(dists[:,1,...]), torch.abs(dists[:,0,...]), torch.abs(dists[:,1,...])))
            local_boundaries = self.normalized_gaussian(dists_B)
            
            wedges_shpd = self.dists2indicators(dists, torch.ones_like(etas[:,:2,:,:])*1e-4)
            patches_shpd = (wedges_shpd.unsqueeze(1) * colors.unsqueeze(-3).unsqueeze(-3)).sum(dim=2)

            sigmas_refoc_1_all = self.depthCal.depth2sigma(depth_1, self.rho_prime)
            sigmas_refoc_2_all = self.depthCal.depth2sigma(depth_2, self.rho_prime)
            sigmas_refoc_1 = torch.where((depth_mask == 1).sum(dim=(1,2)) > 0, sigmas_refoc_1_all, \
                                        torch.ones_like(sigmas_refoc_1_all)*1e-4)
            sigmas_refoc_2 = torch.where((depth_mask == 2).sum(dim=(1,2)) > 0, sigmas_refoc_2_all, \
                                        torch.ones_like(sigmas_refoc_2_all)*1e-4)
            sigmas_refoc = torch.cat([sigmas_refoc_1.unsqueeze(1), sigmas_refoc_2.unsqueeze(1)], dim=1)
            wedges_refoc = self.dists2indicators(dists, sigmas_refoc)
            patches_refoc = (wedges_refoc.unsqueeze(1) * colors.unsqueeze(-3).unsqueeze(-3)).sum(dim=2)

        if colors_only:
            return colors
        else:
            return patches, patches_shpd, patches_refoc, local_boundaries.unsqueeze(1), depth_map, depth_mask

    def forward(self, est, ny_pat, colors_only=True):
        if colors_only:
            est = est.permute(0,2,1).view(self.batch_size*2, 10, self.H_patches, self.W_patches)
            self.img_patches = ny_pat
        else:
            est = est.permute(0,2,1).view(self.batch_size, 12, self.H_patches, self.W_patches)
            self.img_patches = ny_pat.unsqueeze(0)
        xy_angles = est[:,:8,:,:]
        etas = self.params2etas(est[:,8:,:,:])
        if colors_only:
            colors = self.get_patches(xy_angles, etas, colors_only)
            return colors
        else:
            patches, patches_shpd, patches_refoc, local_boundaries, depth_map, depth_mask = self.get_patches(xy_angles, etas, colors_only)
            global_image = self.local2global_color(patches)
            global_image_shpd = self.local2global_color(patches_shpd, pair=False)
            global_image_refoc = self.local2global_color(patches_refoc, pair=False)
            global_bndry = self.local2global_bndry(local_boundaries)
            global_depth_map, confidence_map = self.local2global_depth(depth_map, depth_mask)
            return global_image.detach().cpu().numpy(), global_image_shpd.detach().cpu().numpy(), global_image_refoc.detach().cpu().numpy(), global_bndry.detach().cpu().numpy(), global_depth_map.detach().cpu().numpy(), confidence_map.detach().cpu().numpy()

def generate_raw_depths(args, local_module, global_module, helper, datasetloader, start_idx=180, num_images=20):
    """Generate and save RAW unthresholded depth maps"""
    
    output_dir = './logs/blurry_edges_raw_depths'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("GENERATING RAW UNTHRESHOLDED DEPTH MAPS")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Processing images {start_idx} to {start_idx + num_images - 1}")
    print()

    with torch.no_grad():
        for j, (img_ny, gt_depth) in enumerate(datasetloader):
            if j < start_idx:
                continue
            if j >= start_idx + num_images:
                break
                
            print(f'Processing image #{j}...', end=' ')

            t_img = img_ny.flatten(0,1).permute(0,3,1,2)
            img_patches = nn.Unfold(args.R, stride=args.stride)(t_img).view(2, 3, args.R, args.R, helper.H_patches, helper.W_patches)
            vec = img_patches.permute(0,4,5,1,2,3).reshape(2 * helper.H_patches * helper.W_patches, 3, args.R, args.R)
            params_est = local_module(vec.to(torch.float32))
            params = params_est.view(2, helper.H_patches, helper.W_patches, 10).flatten(start_dim=1,end_dim=2).detach()
            xy = params[:, :, :4]
            angles = torch.remainder(params[:, :, 4:8], 2 * torch.pi)
            etas_coef = params[:, :, 8:]
            params = torch.cat([xy, angles, etas_coef], dim=2)
            colors = helper(params, img_patches, colors_only=True).flatten(start_dim=3,end_dim=4).flatten(start_dim=1,end_dim=2).permute(0,2,1)
            pm = torch.cat([xy / 3, \
                            (angles - torch.pi) / torch.pi, \
                            etas_coef - 0.5, \
                            (colors - 0.5) * 2], dim=2).unsqueeze(0).permute(0,2,1,3).flatten(2,3)

            params = global_module(pm)
            xy = params[:, :, :4] * 3
            angles = torch.remainder((params[:, :, 4:8] + 1) * torch.pi, 2 * torch.pi)
            etas_coef = params[:, :, 8:] + 0.5
            est = torch.cat([xy, angles, etas_coef], dim=2)

            col_est, col_shpd, col_refoc, bndry_est, global_depth_map, confidence_map = helper(est, img_patches, colors_only=False)
            
            # CRITICAL: Save RAW global_depth_map (no thresholding!)
            # This has depth everywhere (100% coverage), but includes noisy low-confidence regions
            np.save(f'{output_dir}/global_depth_raw_{j:03d}.npy', global_depth_map)
            np.save(f'{output_dir}/confidence_{j:03d}.npy', confidence_map)
            
            # Print statistics
            valid_pixels = (confidence_map > 0).sum()
            total_pixels = confidence_map.size
            coverage = valid_pixels / total_pixels * 100
            
            print(f'✓ Saved (coverage: {coverage:.1f}%)')

    print()
    print("="*80)
    print(f"✓ Completed! Saved {num_images} raw depth maps to: {output_dir}")
    print()
    print("These files contain:")
    print("  - global_depth_raw_*.npy: RAW depth everywhere (no threshold)")
    print("  - confidence_*.npy: Confidence maps (same as before)")
    print()
    print("Now you can compare:")
    print("  1. Threshold=0.0 on raw depth (100% coverage, includes noise)")
    print("  2. Neural densifier (100% coverage, learned denoising)")
    print("="*80)

if __name__ == "__main__":
    import sys
    
    # Extract our custom arguments first
    start_idx = 180
    num_images = 20
    
    if '--start_idx' in sys.argv:
        idx = sys.argv.index('--start_idx')
        start_idx = int(sys.argv[idx + 1])
        sys.argv.pop(idx)  # Remove --start_idx
        sys.argv.pop(idx)  # Remove value
    
    if '--num_images' in sys.argv:
        idx = sys.argv.index('--num_images')
        num_images = int(sys.argv[idx + 1])
        sys.argv.pop(idx)  # Remove --num_images
        sys.argv.pop(idx)  # Remove value
    
    args = get_args('eval')
    extra_args = type('Args', (), {'start_idx': start_idx, 'num_images': num_images})()

    device = torch.device(args.cuda)

    dataset_test = TestDataset(device, data_path=args.data_path)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    local_module = LocalStage().to(device)
    local_module.load_state_dict(torch.load(f'{args.model_path}/pretrained_local_stage.pth', map_location=device))
    local_module.eval()
    
    global_module = GlobalStage(in_parameter_size=38, out_parameter_size=12, device=device).to(device)
    if args.densify == 'w':
        global_module.load_state_dict(torch.load(f'{args.model_path}/pretrained_global_stage_w.pth', map_location=device))
    else:
        global_module.load_state_dict(torch.load(f'{args.model_path}/pretrained_global_stage.pth', map_location=device))
    global_module.eval()

    depthCal = DepthEtas(args, device)
    helper = PostProcess(args, depthCal, device)
    
    generate_raw_depths(args, local_module, global_module, helper, test_loader, 
                       start_idx=extra_args.start_idx, num_images=extra_args.num_images)
