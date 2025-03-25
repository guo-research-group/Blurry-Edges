import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import ShapeDataset
from models import GlobalStage
from utils import get_args, set_seed, create_directory, DepthEtas, PostProcessGlobalBase, showCurve, PostProcessBase

class GlobalLoss(PostProcessGlobalBase):
    def __init__(self, args, depthCal, device):
        super().__init__(args, device)
        self.dynamic_epoch = args.dynamic_epoch
        self.gamma_color_range = args.gamma_color
        self.gamma_color_cons_range = args.gamma_color_cons
        self.gamma_bndry_cons_range = args.gamma_bndry_cons
        self.gamma_smthns_range = args.gamma_smthns
        self.gamma_smthns_cons_range = args.gamma_smthns_cons
        self.gamma_bndry_loc_range = args.gamma_bndry_loc
        self.gamma_depth_range = args.gamma_depth
        self.gamma_idx = -1
        self.depthCal = depthCal

    def calculate_gamma(self, gamma_range, rate, order=1):
        return gamma_range[0] + rate**order * (gamma_range[1] - gamma_range[0])

    def update_gamma(self, idx_update=True):
        if idx_update:
            self.gamma_idx += 1

        if self.gamma_idx < self.dynamic_epoch[0]:
            rate = self.gamma_idx / (self.dynamic_epoch[0] - 1)
            case_disp_idx = 0
        elif self.gamma_idx < self.dynamic_epoch[1]:
            rate = 1.0
            case_disp_idx = 0
        elif self.gamma_idx < self.dynamic_epoch[2]:
            rate = (self.gamma_idx - self.dynamic_epoch[1]) / (self.dynamic_epoch[2] - self.dynamic_epoch[1] - 1)
            case_disp_idx = 1
        else:
            rate = 1.0
            case_disp_idx = 1

        self.gamma_color = self.calculate_gamma(self.gamma_color_range[0+case_disp_idx:2+case_disp_idx], rate)
        self.gamma_color_cons = self.calculate_gamma(self.gamma_color_cons_range[0+case_disp_idx:2+case_disp_idx], rate)
        self.gamma_bndry_cons = self.calculate_gamma(self.gamma_bndry_cons_range[0+case_disp_idx:2+case_disp_idx], rate)
        self.gamma_smthns = self.calculate_gamma(self.gamma_smthns_range[0+case_disp_idx:2+case_disp_idx], rate)
        self.gamma_smthns_cons = self.calculate_gamma(self.gamma_smthns_cons_range[0+case_disp_idx:2+case_disp_idx], rate)
        self.gamma_bndry_loc = self.calculate_gamma(self.gamma_bndry_loc_range[0+case_disp_idx:2+case_disp_idx], rate)
        self.gamma_depth = self.calculate_gamma(self.gamma_depth_range[0+case_disp_idx:2+case_disp_idx], rate)

    def final_gamma(self):
        self.gamma_color = self.gamma_color_range[-1]
        self.gamma_color_cons = self.gamma_color_cons_range[-1]
        self.gamma_bndry_cons = self.gamma_bndry_cons_range[-1]
        self.gamma_smthns = self.gamma_smthns_range[-1]
        self.gamma_smthns_cons = self.gamma_smthns_cons_range[-1]
        self.gamma_bndry_loc = self.gamma_bndry_loc_range[-1]
        self.gamma_depth = self.gamma_depth_range[-1]

    def get_colors(self, wedges, img_patches):
        A = wedges.permute(0,5,6,1,3,4,2).reshape(self.batch_size, self.H_patches, self.W_patches, -1, 3)
        A_t = A.permute(0,1,2,4,3)
        y = img_patches.permute(0,5,6,1,3,4,2).reshape(self.batch_size, self.H_patches, self.W_patches, -1, 3)
        colors = torch.matmul(self.inverse_3by3(torch.matmul(A_t, A)+self.ridge), torch.matmul(A_t, y)).permute(0,4,3,1,2)
        return colors

    def get_patches(self, xy_angles, etas):
        dists = self.params2dists(xy_angles)
        
        wedges1 = self.dists2indicators(dists, etas[:,:2,:,:])
        wedges2 = self.dists2indicators(dists, etas[:,2:,:,:])
        wedges = torch.cat([wedges1.unsqueeze(1), wedges2.unsqueeze(1)], dim=1)
        colors = self.get_colors(wedges, self.img_patches)
        patches1 = (wedges1.unsqueeze(1) * colors.unsqueeze(-3).unsqueeze(-3)).sum(dim=2)
        patches2 = (wedges2.unsqueeze(1) * colors.unsqueeze(-3).unsqueeze(-3)).sum(dim=2)
        patches = torch.cat([patches1.unsqueeze(1), patches2.unsqueeze(1)], dim=1)

        dists_B = torch.where(dists[:,1,...] >= 0, dists[:,1,...], \
                              torch.where(torch.abs(dists[:,0,...])<torch.abs(dists[:,1,...]), torch.abs(dists[:,0,...]), torch.abs(dists[:,1,...])))
        local_boundaries = self.normalized_gaussian(dists_B)

        depth_mask = (self.normalized_gaussian(dists[:,0,...]) > 0.5).to(torch.int32)
        depth_mask_temp = (self.normalized_gaussian(dists[:,1,...]) > 0.5).to(torch.int32) * 2
        depth_mask = torch.where((depth_mask_temp == 2) | (dists[:,1,...] >= 0), depth_mask_temp, depth_mask)
        depth_1 = self.depthCal.etas2depth(etas[:,0,:,:], etas[:,2,:,:])
        depth_2 = self.depthCal.etas2depth(etas[:,1,:,:], etas[:,3,:,:])
        depth_map = torch.where(depth_mask == 1, depth_1.unsqueeze(1).unsqueeze(1), \
                                torch.where(depth_mask == 2, depth_2.unsqueeze(1).unsqueeze(1), depth_mask))
        return patches, local_boundaries.unsqueeze(1), depth_map, depth_mask

    def get_color_consistency_term(self, patches):
        curr_global_image_patches = nn.Unfold(self.R, stride=self.stride)( \
            self.global_image.view(self.batch_size * 2, 3, self.H, self.W).detach()).view(self.batch_size, 2, 3, self.R, self.R, self.H_patches, self.W_patches)
        consistency = ((patches - curr_global_image_patches) ** 2).sum(2)
        return consistency

    def get_boundary_consistency_term(self, local_boundaries):
        curr_global_boundaries_patches = nn.Unfold(self.R, stride=self.stride)( \
            self.global_bndry.detach()).view(self.batch_size, 1, self.R, self.R, self.H_patches, self.W_patches)
        consistency = ((local_boundaries - curr_global_boundaries_patches) ** 2).squeeze(1)
        return consistency

    def get_smoothness_term(self, patches, global_image_deri_gt):
        global_image_deri = self.get_image_derivative(self.global_image.view(self.batch_size * 2, 3, self.H, self.W).detach())
        curr_global_image_deri_patches = nn.Unfold(self.R-2, stride=self.stride)( \
            global_image_deri).view(self.batch_size, 2, 3, self.R-2, self.R-2, self.H_patches, self.W_patches)
        curr_global_image_deri_patches_gt = nn.Unfold(self.R-2, stride=self.stride)( \
            global_image_deri_gt.permute(0,1,4,2,3).view(self.batch_size * 2, 3, self.H-2, self.W-2)).view(self.batch_size, 2, 3, self.R-2, self.R-2, self.H_patches, self.W_patches)
        patches_deri = self.get_image_derivative(patches.permute(0,1,5,6,2,3,4).flatten(start_dim=0,end_dim=3)).view(self.batch_size, 2, self.H_patches, self.W_patches, 3, self.R-2, self.R-2).permute(0,1,4,5,6,2,3)
        smthns_error = ((patches_deri - curr_global_image_deri_patches_gt) ** 2).sum(2)
        smthns_cons_error = ((patches_deri - curr_global_image_deri_patches) ** 2).sum(2)
        return smthns_error, smthns_cons_error

    def get_boundary_localization_term(self, local_boundaries, bndry_dist):
        curr_global_bndry_dist_patches = nn.Unfold(self.R, stride=self.stride)( \
            torch.log2(bndry_dist+1)).view(self.batch_size, 1, self.R, self.R, self.H_patches, self.W_patches)
        loc_error = ((curr_global_bndry_dist_patches * local_boundaries) ** 2).squeeze(1)
        return loc_error

    def get_depth_term(self, depth_map, bndry_depth, depth_mask):
        curr_global_bndry_depth_patches = nn.Unfold(self.R, stride=self.stride)( \
            bndry_depth).view(self.batch_size, self.R, self.R, self.H_patches, self.W_patches)
        mask = torch.where(curr_global_bndry_depth_patches == 0, torch.zeros_like(curr_global_bndry_depth_patches), \
                           torch.where(depth_mask == 0, torch.zeros_like(curr_global_bndry_depth_patches), torch.ones_like(curr_global_bndry_depth_patches)))
        depth_error = (((depth_map - curr_global_bndry_depth_patches) * mask) ** 2).sum() / mask.sum()
        return depth_error

    def get_loss(self, local_boundaries, patches, bndry_dist, deri, depth_map, bndry_depth, depth_mask):
        smthns_error, smthns_cons_error = self.get_smoothness_term(patches, deri)
        loss_per_patch = self.gamma_color * ((self.gt_img_patches - patches) ** 2).sum(2).mean() + \
                            self.gamma_color_cons * self.get_color_consistency_term(patches).mean() + \
                            self.gamma_bndry_cons * self.get_boundary_consistency_term(local_boundaries).mean() + \
                            self.gamma_smthns * smthns_error.mean() + \
                            self.gamma_smthns_cons * smthns_cons_error.mean() + \
                            self.gamma_bndry_loc * self.get_boundary_localization_term(local_boundaries, bndry_dist.unsqueeze(1)).mean() + \
                            self.gamma_depth * self.get_depth_term(depth_map, bndry_depth.unsqueeze(1), depth_mask)
        return loss_per_patch

    def split_restore_params(self, est):
        xy = est[:,:4,:,:] * 3
        angles = torch.remainder((est[:,4:8,:,:] + 1) * torch.pi, 2 * torch.pi)
        etas = self.params2etas(est[:,8:,:,:] + 0.5)
        return torch.cat([xy, angles], dim=1), etas

    def forward(self, est, img_ny, img_gt, bndry_dist, deri, bndry_depth):
        est = est.permute(0,2,1).view(self.batch_size, 12, self.H_patches, self.W_patches)
        xy_angles, etas = self.split_restore_params(est)

        self.img_patches = nn.Unfold(self.R, stride=self.stride)(img_ny.reshape(self.batch_size*2,self.H,self.W,3).permute(0,3,1,2)).view(self.batch_size, 2, 3, self.R, self.R, self.H_patches, self.W_patches)
        self.gt_img_patches = nn.Unfold(self.R, stride=self.stride)(img_gt.reshape(self.batch_size*2,self.H,self.W,3).permute(0,3,1,2)).view(self.batch_size, 2, 3, self.R, self.R, self.H_patches, self.W_patches)
        patches, local_boundaries, depth_map, depth_mask = self.get_patches(xy_angles, etas)
        self.global_image = self.local2global_color(patches)
        self.global_bndry = self.local2global_bndry(local_boundaries)
        loss = self.get_loss(local_boundaries, patches, bndry_dist, deri, depth_map, bndry_depth, depth_mask)
        return loss

def evaluateDataset(args, model, criteria, datasetloader, data_size):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        criteria.final_gamma()
        for param, img_ny, img_gt, bndry_dist, deri, bndry_depth in datasetloader:
            est = model(param.permute(0,2,1,3).flatten(2,3))
            loss = criteria(est, img_ny, img_gt, bndry_dist, deri, bndry_depth)
            total_loss += loss
        criteria.update_gamma(idx_update=False)
        num_batch = data_size // args.batch_size
        avg_total_loss = total_loss / num_batch
        return avg_total_loss

if __name__ == "__main__":
    
    args = get_args('global_train')
    
    set_seed(1898, deterministic=True)
    create_directory(args.log_path, overwrite=False)
    
    device = torch.device(args.cuda)
    dataset_train = ShapeDataset(device, data_path=args.data_path, train=True, mode='global')
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dataset_val = ShapeDataset(device, data_path=args.data_path, train=False, mode='global')
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=True)

    estimator_global = GlobalStage(in_parameter_size=args.input_size, out_parameter_size=args.output_size, device=device).to(device)
    for p in estimator_global.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
    optimizer = torch.optim.AdamW(estimator_global.parameters(), lr=args.learning_rate)
    depthCal = DepthEtas(args, device)
    criteria = GlobalLoss(args, depthCal, device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.975, patience=5, min_lr=args.learning_rate*0.5)

    lr_updated = 0
    best_avg_loss = np.inf
    best_epoch = 0
    avg_total_loss = np.zeros((args.epoch_num,), dtype=float)
    f = open(f'{args.log_path}/exp_global_stage_training.txt', 'wt')
    print('Arguments:', file=f, flush=True)
    for arg in vars(args):
        print(f'{arg:<20}: {getattr(args, arg)}', file=f, flush=True)
    print('\nTraining:', file=f, flush=True)
    print(f'{"Epoch":<10} {"Loss":<20} {"Scheduler patience":<20} {"Learning rate"}', file=f, flush=True)
    for epoch in tqdm(range(args.epoch_num)):
        criteria.update_gamma()
        estimator_global.train()
        for step, (param, _, img_gt, bndry_dist, deri, bndry_depth) in enumerate(train_loader):
            est = estimator_global(param.permute(0,2,1,3).flatten(2,3))
            optimizer.zero_grad()
            loss = criteria(est, img_gt, img_gt, bndry_dist, deri, bndry_depth)
            loss.backward()
            nn.utils.clip_grad_norm_(estimator_global.parameters(), max_norm=1, norm_type=2)
            optimizer.step()
        avg_total_loss[epoch] = evaluateDataset(args, estimator_global, criteria, val_loader, len(dataset_val))
        print(f'{epoch+1:<10} {avg_total_loss[epoch]:<20.10f} {scheduler.patience:<20} {optimizer.param_groups[0]["lr"]:.4e}', file=f, flush=True)
        if avg_total_loss[epoch] < best_avg_loss:
            best_avg_loss = avg_total_loss[epoch]
            torch.save(estimator_global.state_dict(), f'{args.model_path}/best_run_exp_global_stage.pth')
            best_epoch = epoch
        if epoch >= args.dynamic_epoch[1]:
            scheduler.step(avg_total_loss[epoch])
    np.save(f'{args.log_path}/loss_curve_exp_global_stage.npy', avg_total_loss)
    showCurve(args, avg_total_loss, 'loss_curve_exp_global_stage')
    print(f'\n-- Best epoch is the {best_epoch+1:d}th, with average loss of {best_avg_loss:.10f}.', file=f, flush=True)
    f.close()