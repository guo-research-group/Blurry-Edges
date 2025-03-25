import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import ShapeDataset
from models import LocalStage
from utils import get_args, PostProcessLocalBase

def ref_data_gen(args, model, helper, datasetloader, partition):
    H_patches = int(np.floor((args.img_size[0] - args.R) / args.stride) + 1)
    W_patches = int(np.floor((args.img_size[1] - args.R) / args.stride) + 1)
    params_src = np.zeros((len(datasetloader), 2, H_patches*W_patches, 19), dtype=np.float64)
    with torch.no_grad():
        for j, ny_img in tqdm(enumerate(datasetloader), total=len(datasetloader)):
            t_img = ny_img.flatten(0,1).permute(0,3,1,2)
            img_patches = nn.Unfold(args.R, stride=args.stride)(t_img).view(2, 3, args.R, args.R, H_patches, W_patches)
            vec = img_patches.permute(0,4,5,1,2,3).reshape(2*H_patches*W_patches, 3, args.R, args.R)
            params_est = model(vec)
            params = params_est.view(2*H_patches*W_patches, 10).detach()
            xy = params[:, :4]
            angles = torch.remainder(params[:, 4:8], 2 * torch.pi)
            etas_coef = params[:, 8:]

            params = torch.cat([xy, angles, etas_coef], dim=1)
            colors = helper(params, vec.permute(0,2,3,1)).flatten(start_dim=1,end_dim=2)
            pm = torch.cat([xy / 3, \
                            (angles - torch.pi) / torch.pi, \
                            etas_coef - 0.5, \
                            (colors - 0.5) * 2], dim=1).view(2, H_patches*W_patches, 19)
            params_src[j, :, :, :] = pm.detach().cpu().numpy()

    np.save(f'{args.data_path}/params_src_{partition}.npy', params_src)

class PostProcess(PostProcessLocalBase):
    def __init__(self, args, device):
        super().__init__(args, device)
    
    def get_colors(self, params, pat_ny):
        dists = self.params2dists(params[:, :8])
        etas = self.params2etas(params[:, 8:])
        wedges = self.dists2indicators(dists, etas)
        A = wedges.permute(0,2,3,1).flatten(start_dim=1,end_dim=2)
        A_t = A.permute(0,2,1)
        y = pat_ny.flatten(start_dim=1,end_dim=2)
        colors = torch.matmul(self.inverse_3by3(torch.matmul(A_t, A)+self.ridge), torch.matmul(A_t, y)).permute(0,2,1)
        return colors

    def forward(self, params, pat_ny):
        return self.get_colors(params, pat_ny)

if __name__ == "__main__":

    args = get_args('global_pre')

    device = torch.device(args.cuda)

    dataset_train = ShapeDataset(device, data_path=args.data_path, train=True, mode='global_pre')
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False)
    dataset_val = ShapeDataset(device, data_path=args.data_path, train=False, mode='global_pre')
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)

    local_module = LocalStage().to(device)
    local_module.load_state_dict(torch.load(f'{args.model_path}/pretrained_local_stage.pth')) # change the path to your local stage weights
    local_module.eval()
    helper = PostProcess(args, device)
    print('Processing training set...')
    ref_data_gen(args, local_module, helper, train_loader, 'train')
    print('Processing validation set...')
    ref_data_gen(args, local_module, helper, val_loader, 'val')