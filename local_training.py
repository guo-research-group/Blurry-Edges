import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from data import ShapeDataset
from models import LocalStage
from utils import get_args, set_seed, create_directory, PostProcessLocalBase, showCurve

class LocalLoss(PostProcessLocalBase):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.max_beta_bndry_loc = args.beta_bndry_loc
        self.max_beta_smthns = args.beta_smthns
        self.beta_idx = -1
        self.dynamic_epoch = args.dynamic_epoch

    def update_beta(self, idx_update=True):
        if idx_update:
            self.beta_idx += 1
        if self.beta_idx < self.dynamic_epoch:
            rate = self.beta_idx / (self.dynamic_epoch - 1)
        else:
            rate = 1.0
        self.beta_bndry_loc = rate * self.max_beta_bndry_loc
        self.beta_smthns = rate * self.max_beta_smthns
    
    def final_beta(self):
        self.beta_bndry_loc = self.max_beta_bndry_loc
        self.beta_smthns = self.max_beta_smthns

    def get_patches(self, est, img_ny):
        est[:, 4:8] = torch.remainder(est[:, 4:8], 2 * torch.pi)
        dists = self.params2dists(est[:, :8])
        etas = self.params2etas(est[:, 8:])
        wedges = self.dists2indicators(dists, etas)
        A = wedges.permute(0,2,3,1).view(self.batch_size, -1, 3)
        A_t = A.permute(0,2,1)
        y = img_ny.view(self.batch_size, -1, 3)
        colors = torch.matmul(self.inverse_3by3(torch.matmul(A_t, A)+self.ridge), torch.matmul(A_t, y)).permute(0,2,1)
        patches = (wedges.unsqueeze(1) * colors.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)
        dists_B = torch.where(dists[:,1,:,:] >= 0, dists[:,1,:,:], \
                              torch.where(torch.abs(dists[:,0,:,:])<torch.abs(dists[:,1,:,:]), torch.abs(dists[:,0,:,:]), torch.abs(dists[:,1,:,:])))
        local_boundaries = self.normalized_gaussian(dists_B)
        return patches, local_boundaries

    def forward(self, est, img_ny, gt_img, bndry_dist, deri):
        patches, local_boundaries = self.get_patches(est, img_ny)
        loss = ((gt_img - patches.permute(0,2,3,1)) ** 2).sum(-1).mean() \
                + self.beta_bndry_loc * ((bndry_dist * local_boundaries) ** 2).mean() \
                + self.beta_smthns * ((deri.permute(0,3,1,2) - self.get_image_derivative(patches)) ** 2).sum(1).mean()
        return loss

def evaluateDataset(args, criteria, model, datasetloader, data_size):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        criteria.final_beta()
        for img_ny, img_gt, bndry_dist, deri in datasetloader:
            est = model(img_ny.permute(0,3,1,2))
            loss = criteria(est, img_ny, img_gt, bndry_dist, deri)
            total_loss += loss
        criteria.update_beta(idx_update=False)
        num_batch = data_size // args.batch_size
        avg_total_loss = total_loss / num_batch
        return avg_total_loss

if __name__ == "__main__":

    args = get_args('local_train')
    
    # export CUBLAS_WORKSPACE_CONFIG=:4096:8
    set_seed(1869)
    create_directory(args.log_path, overwrite=False)

    device = torch.device(args.cuda)
    dataset_train = ShapeDataset(device, data_path=args.data_path, train=True)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dataset_val = ShapeDataset(device, data_path=args.data_path, train=False)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=True)

    estimator_local = LocalStage().to(device)
    for p in estimator_local.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
    optimizer = torch.optim.AdamW(estimator_local.parameters(), lr=args.learning_rate)
    criteria = LocalLoss(args, device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=2, min_lr=args.learning_rate*0.1)

    lr_updated = 0
    best_avg_loss = np.inf
    best_epoch = 0
    avg_total_loss = np.zeros((args.epoch_num,), dtype=float)
    f = open(f'{args.log_path}/exp_local_stage_training.txt', 'wt')
    print('Arguments:', file=f, flush=True)
    for arg in vars(args):
        print(f'{arg:<20}: {getattr(args, arg)}', file=f, flush=True)
    print('\nTraining:', file=f, flush=True)
    print(f'{"Epoch":<10} {"Loss":<20} {"Scheduler patience":<20} {"Learning rate"}', file=f, flush=True)
    for epoch in tqdm(range(args.epoch_num)):
        criteria.update_beta()
        estimator_local.train()
        for step, (img_ny, img_gt, bndry_dist, deri) in enumerate(train_loader):
            est = estimator_local(img_ny.permute(0,3,1,2))
            optimizer.zero_grad()
            loss = criteria(est, img_gt, img_gt, bndry_dist, deri)
            loss.backward()
            nn.utils.clip_grad_norm_(estimator_local.parameters(), max_norm=1, norm_type=2)
            optimizer.step()
        avg_total_loss[epoch] = evaluateDataset(args, criteria, estimator_local, val_loader, len(dataset_val))

        scheduler.step(avg_total_loss[epoch])
        scheduler.patience = 2 + int(np.log2(epoch+1)) * 3
        
        print(f'{epoch+1:<10} {avg_total_loss[epoch]:<20.10f} {scheduler.patience:<20} {optimizer.param_groups[0]["lr"]:.4e}', file=f, flush=True)
        if avg_total_loss[epoch] < best_avg_loss:
            best_avg_loss = avg_total_loss[epoch]
            torch.save(estimator_local.state_dict(), f'{args.model_path}/best_run_exp_local_stage.pth')
            best_epoch = epoch
    np.save(f'{args.log_path}/loss_curve_exp_local_stage.npy', avg_total_loss)
    showCurve(args, avg_total_loss, 'loss_curve_exp_local_stage')
    print(f'\n-- Best epoch is the {best_epoch+1:d}th, with average loss of {best_avg_loss:.10f}', file=f, flush=True)
    f.close()