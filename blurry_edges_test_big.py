import numpy as np
import os
import cv2
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import TestDataset
from models import LocalStage, GlobalStage
from utils import get_args, DepthEtas, PostProcessGlobalBase, Visualizer, eval_depth

class PostProcess(PostProcessGlobalBase):
    def __init__(self, args, depthCal, device):
        super().__init__(args, device)
        self.depthCal = depthCal
        self.rho_prime = args.rho_prime
    
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

            depth_mask = (self.normalized_gaussian(dists[:,0,...]) > 0.5).to(torch.int32)
            depth_mask_temp = (self.normalized_gaussian(dists[:,1,...]) > 0.5).to(torch.int32) * 2
            depth_mask = torch.where((depth_mask_temp == 2) | (dists[:,1,...] >= 0), depth_mask_temp, depth_mask)
            depth_1 = self.depthCal.etas2depth(etas[:,0,:,:], etas[:,2,:,:])
            depth_2 = self.depthCal.etas2depth(etas[:,1,:,:], etas[:,3,:,:])
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
            return patches, patches_shpd, patches_refoc, local_boundaries, depth_map, depth_mask

def local2global_color(args, num_patches, patches, pair=True):
    if pair:
        return (torch.nn.Fold(output_size=[args.big_img_size[0], args.big_img_size[1]], kernel_size=args.R, stride=args.stride)(
                patches.view(args.batch_size*2, 3*args.R**2, -1)).view(args.batch_size, 2, 3, args.big_img_size[0], args.big_img_size[1]) / \
                num_patches.unsqueeze(0).unsqueeze(0).unsqueeze(0)).detach().cpu().numpy()
    else:
        return (torch.nn.Fold(output_size=[args.big_img_size[0], args.big_img_size[1]], kernel_size=args.R, stride=args.stride)(
                patches.view(args.batch_size, 3*args.R**2, -1)).view(args.batch_size, 3, args.big_img_size[0], args.big_img_size[1]) / \
                num_patches.unsqueeze(0).unsqueeze(0)).detach().cpu().numpy()

def local2global_bndry(args, num_patches, bndry_patches):
    return (torch.nn.Fold(output_size=[args.big_img_size[0], args.big_img_size[1]], kernel_size=args.R, stride=args.stride)(
            bndry_patches.view(args.batch_size, args.R**2, -1)).view(args.batch_size, 1, args.big_img_size[0], args.big_img_size[1]) / \
            num_patches.unsqueeze(0).unsqueeze(0)).detach().cpu().numpy()

def local2global_depth(args, H_patches, W_patches, num_patches, depth_map, depth_mask):
    num_depth_patches = torch.nn.Fold(output_size=[args.big_img_size[0], args.big_img_size[1]], kernel_size=args.R, stride=args.stride)(
                        (depth_mask.view(args.batch_size,args.R**2,H_patches*W_patches) > 0).to(torch.float32)).view(args.batch_size, args.big_img_size[0], args.big_img_size[1])
    confidence_map = num_depth_patches / num_patches.unsqueeze(0)
    depth_map = torch.nn.Fold(output_size=[args.big_img_size[0], args.big_img_size[1]], kernel_size=args.R, stride=args.stride)(
                depth_map.view(args.batch_size, args.R**2, -1)).view(args.batch_size, args.big_img_size[0], args.big_img_size[1]) / \
                torch.where(num_depth_patches > 0, num_depth_patches, torch.ones_like(num_depth_patches))
    return depth_map.detach().cpu().numpy(), confidence_map.detach().cpu().numpy()

def depth_estimator(args, local_module, global_module, helper, visualizer, datasetloader):
    total_delta1, total_delta2, total_delta3, total_RMSE, total_AbsRel = 0, 0, 0, 0, 0
    total_running_time = 0
    block_stride = tuple((np.array(args.img_size) - args.R + args.stride - args.stride*args.n_margin_patch*2).astype(np.int64))
    n_block = tuple(np.ceil((np.array(args.big_img_size) - args.R  - args.stride*args.n_margin_patch*2 + args.stride) / np.array(block_stride)).astype(np.int64))

    H_patches = int(np.floor((args.big_img_size[0] - args.R) / args.stride) + 1)
    W_patches = int(np.floor((args.big_img_size[1] - args.R) / args.stride) + 1)
    num_patches = torch.nn.Fold(output_size=[args.big_img_size[0], args.big_img_size[1]],
                                kernel_size=args.R,
                                stride=args.stride)(torch.ones(1, args.R**2,
                                                                H_patches * W_patches,
                                                                device=helper.device)).view(*args.big_img_size)

    if not os.path.exists(f'{args.log_path}/visualizations_big/'):
        os.makedirs(f'{args.log_path}/visualizations_big/')

    with torch.no_grad():
        for j, (img_ny, gt_depth) in enumerate(datasetloader):
            print(f'Image pair #{j}:')
            start_time = time.time()

            blocks = nn.Unfold(args.img_size, stride=block_stride)(img_ny.flatten(0,1).permute(0,3,1,2)).view(2, 3, *args.img_size, *n_block)
            full_patches = torch.zeros((2, 3, args.R, args.R, H_patches,  W_patches)).to(helper.device)
            full_patches_shpd = torch.zeros((3, args.R, args.R, H_patches,  W_patches)).to(helper.device)
            full_patches_refoc = torch.zeros((3, args.R, args.R, H_patches,  W_patches)).to(helper.device)
            full_local_boundaries = torch.zeros((args.R, args.R, H_patches,  W_patches)).to(helper.device)
            full_depth_map = torch.zeros((args.R, args.R, H_patches,  W_patches)).to(helper.device)
            full_depth_mask = torch.zeros((args.R, args.R, H_patches,  W_patches)).to(helper.device)
            for i_v in range(n_block[0]):
                for i_h in range(n_block[1]):
                    t_img = blocks[..., i_v, i_h]
                    img_patches = nn.Unfold(args.R, stride=args.stride)(t_img).view(2, 3, args.R, args.R, helper.H_patches, helper.W_patches)
                    vec = img_patches.permute(0,4,5,1,2,3).reshape(2*helper.H_patches*helper.W_patches, 3, args.R, args.R)
                    params_est = local_module(vec.to(torch.float32))
                    params = params_est.view(2, helper.H_patches, helper.W_patches, 10).flatten(start_dim=1,end_dim=2).detach()
                    xy = params[..., :4]
                    angles = torch.remainder(params[..., 4:8], 2 * torch.pi)
                    etas_coef = params[..., 8:]
                    params = torch.cat([xy, angles, etas_coef], dim=-1)
                    colors = helper(params, img_patches, colors_only=True).flatten(start_dim=3,end_dim=4).flatten(start_dim=1,end_dim=2).permute(0,2,1)
                    pm = torch.cat([xy / 3, \
                                    (angles - torch.pi) / torch.pi, \
                                    etas_coef - 0.5, \
                                    (colors - 0.5) * 2], dim=-1).unsqueeze(0).permute(0,2,1,3).flatten(2,3)

                    params = global_module(pm)
                    xy = params[:, :, :4] * 3
                    angles = torch.remainder((params[:, :, 4:8] + 1) * torch.pi, 2 * torch.pi)
                    etas_coef = params[:, :, 8:] + 0.5
                    est = torch.cat([xy, angles, etas_coef], dim=-1)

                    patches, patches_shpd, patches_refoc, local_boundaries, depth_map, depth_mask = helper(est, img_patches, colors_only=False)
                    v_s = 1 if i_v == 0 else 0
                    v_e = 1 if i_v == n_block[0]-1 else 0
                    h_s = 1 if i_h == 0 else 0
                    h_e = 1 if i_h == n_block[1]-1 else 0
                    V_s = i_v*(helper.H_patches-2*args.n_margin_patch)+(1-v_s)*args.n_margin_patch
                    V_e = (i_v+1)*(helper.H_patches-2*args.n_margin_patch)+(1+v_e)*args.n_margin_patch
                    H_s = i_h*(helper.W_patches-2*args.n_margin_patch)+(1-h_s)*args.n_margin_patch
                    H_e = (i_h+1)*(helper.W_patches-2*args.n_margin_patch)+(1+h_e)*args.n_margin_patch
                    V_s_l = (1-v_s)*args.n_margin_patch
                    V_e_l = (v_e-1)*args.n_margin_patch+helper.H_patches
                    H_s_l = (1-h_s)*args.n_margin_patch
                    H_e_l = (h_e-1)*args.n_margin_patch+helper.W_patches
                    full_patches[:,:,:,:,V_s:V_e,H_s:H_e] = patches.clone()[0,:,:,:,:,V_s_l:V_e_l,H_s_l:H_e_l]
                    full_patches_shpd[:,:,:,V_s:V_e,H_s:H_e] = patches_shpd.clone()[0,:,:,:,V_s_l:V_e_l,H_s_l:H_e_l]
                    full_patches_refoc[:,:,:,V_s:V_e,H_s:H_e] = patches_refoc.clone()[0,:,:,:,V_s_l:V_e_l,H_s_l:H_e_l]
                    full_local_boundaries[:,:,V_s:V_e,H_s:H_e] = local_boundaries.clone()[0,0,:,:,V_s_l:V_e_l,H_s_l:H_e_l]
                    full_depth_map[:,:,V_s:V_e,H_s:H_e] = depth_map.clone()[0,:,:,V_s_l:V_e_l,H_s_l:H_e_l]
                    full_depth_mask[:,:,V_s:V_e,H_s:H_e] = depth_mask.clone()[0,:,:,V_s_l:V_e_l,H_s_l:H_e_l]

            col_est = local2global_color(args, num_patches, full_patches)
            col_shpd = local2global_color(args, num_patches, full_patches_shpd, pair=False)
            col_refoc = local2global_color(args, num_patches, full_patches_refoc, pair=False)
            bndry_est = local2global_bndry(args, num_patches, full_local_boundaries)
            global_depth_map, confidence_map = local2global_depth(args, H_patches, W_patches, num_patches, full_depth_map, full_depth_mask)
            depth_map = np.where(confidence_map > 0.05, global_depth_map, np.zeros_like(global_depth_map))

            running_time = time.time() - start_time
            total_running_time += running_time

            error_mask = depth_map>0.0
            delta1, delta2, delta3, RMSE, AbsRel = eval_depth(depth_map, gt_depth.detach().cpu().numpy(), error_mask, crop=args.crop)
            total_delta1 += delta1
            total_delta2 += delta2
            total_delta3 += delta3
            total_RMSE += RMSE
            total_AbsRel += AbsRel
            print(f'--- Error metrics: delta1 ={delta1: .3f}, delta2 ={delta2: .3f}, delta3 ={delta3: .3f}, RMSE ={RMSE: .3f} cm, AbsRel ={AbsRel: .3f} cm')

            result_plot = visualizer.visualize(img_ny.detach().cpu().numpy()[0,0,:,:,:],
                                               img_ny.detach().cpu().numpy()[0,1,:,:,:],
                                               col_est[0,0,:,:,:].transpose(1, 2, 0),
                                               col_est[0,1,:,:,:].transpose(1, 2, 0),
                                               col_shpd[0,:,:,:].transpose(1, 2, 0),
                                               col_refoc[0,:,:,:].transpose(1, 2, 0),
                                               confidence_map[0,:,:],
                                               bndry_est[0,0,:,:],
                                               gt_depth[0,:,:].detach().cpu().numpy(),
                                               depth_map[0,:,:])
            cv2.imwrite(f'{args.log_path}/visualizations_big/{j}.png', result_plot)

            print(f'--- Running time:{running_time: .3f} s')

        n_sample = len(datasetloader)
        print(f'\nAverage running time:{total_running_time/n_sample: .3f} s')
        print(f'Average metrics for whole dataset: delta1 ={total_delta1/n_sample: .3f}, delta2 ={total_delta2/n_sample: .3f}, delta3 ={total_delta3/n_sample: .3f}, RMSE ={total_RMSE/n_sample: .3f} cm, AbsRel ={total_AbsRel/n_sample: .3f} cm')

if __name__ == "__main__":
    args = get_args('eval', big=True)

    device = torch.device(args.cuda)

    dataset_test = TestDataset(device, data_path=args.data_path)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    local_module = LocalStage().to(device)
    local_module.load_state_dict(torch.load(f'{args.model_path}/pretrained_local_stage.pth')) # change the path to your local stage weights
    local_module.eval()
    
    global_module = GlobalStage(in_parameter_size=38, out_parameter_size=12, device=device).to(device)
    global_module.load_state_dict(torch.load(f'{args.model_path}/pretrained_global_stage.pth')) # change the path to your global stage weights
    global_module.eval()

    depthCal = DepthEtas(args, device)
    helper = PostProcess(args, depthCal, device)
    visualizer = Visualizer(args.rho_prime, img_size=args.big_img_size[0], scale=2)
    depth_estimator(args, local_module, global_module, helper, visualizer, test_loader)