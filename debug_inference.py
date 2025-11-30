import torch
import numpy as np
import torch.nn as nn
from models import LocalStage, GlobalStage
from utils import get_args, DepthEtas
from data import TestDataset

args = get_args('eval')
args.data_path = './data_test/regular'
args.cuda = 'cpu'

device = torch.device(args.cuda)

# Load one sample
dataset_test = TestDataset(device, data_path=args.data_path)
img_ny, gt_depth = dataset_test[0]

print("="*50)
print("SAMPLE DATA INSPECTION")
print("="*50)
print(f"img_ny shape: {img_ny.shape}")
print(f"img_ny range: [{img_ny.min():.2f}, {img_ny.max():.2f}]")
print(f"img_ny mean: {img_ny.mean():.2f}")
print(f"gt_depth shape: {gt_depth.shape}")
print(f"gt_depth range: [{gt_depth.min():.4f}, {gt_depth.max():.4f}]")

# Extract patches like in the test code  
img_ny_batch = img_ny.unsqueeze(0)  # Add batch dimension
t_img = img_ny_batch.flatten(0,1).permute(0,3,1,2)
print(f"\nt_img shape after flatten & permute: {t_img.shape}")
print(f"t_img range: [{t_img.min():.2f}, {t_img.max():.2f}]")

img_patches = nn.Unfold(args.R, stride=args.stride)(t_img).view(2, 3, args.R, args.R, -1, -1)
print(f"img_patches shape: {img_patches.shape}")
print(f"img_patches range: [{img_patches.min():.2f}, {img_patches.max():.2f}]")

H_patches = img_patches.shape[4]
W_patches = img_patches.shape[5]
print(f"Number of patches: H={H_patches}, W={W_patches}, Total={H_patches*W_patches}")

vec = img_patches.permute(0,4,5,1,2,3).reshape(2 * H_patches * W_patches, 3, args.R, args.R)
print(f"\nvec shape (input to local model): {vec.shape}")
print(f"vec range: [{vec.min():.2f}, {vec.max():.2f}]")
print(f"vec dtype: {vec.dtype}")

# Load and test local model
local_module = LocalStage().to(device)
local_module.load_state_dict(torch.load(f'{args.model_path}/pretrained_local_stage.pth', map_location=device, weights_only=False))
local_module.eval()

print("\n" + "="*50)
print("LOCAL MODEL PREDICTION")
print("="*50)
with torch.no_grad():
    # Test on first few patches
    test_vec = vec[:10]
    print(f"Testing on {test_vec.shape[0]} patches...")
    params_est = local_module(test_vec.to(torch.float32))
    print(f"params_est shape: {params_est.shape}")
    print(f"params_est range: [{params_est.min():.2f}, {params_est.max():.2f}]")
    print(f"params_est mean per channel:")
    for i in range(10):
        print(f"  Channel {i}: mean={params_est[:, i].mean():.4f}, std={params_est[:, i].std():.4f}")
    
    # Full prediction
    print(f"\nRunning full prediction on all {vec.shape[0]} patches...")
    params_est_full = local_module(vec.to(torch.float32))
    print(f"Full params_est shape: {params_est_full.shape}")
    print(f"Full params_est range: [{params_est_full.min():.2f}, {params_est_full.max():.2f}]")
    
    # Reshape as in test code
    params = params_est_full.view(2, H_patches, W_patches, 10).flatten(start_dim=1,end_dim=2).detach()
    print(f"\nReshaped params shape: {params.shape}")
    xy = params[:, :, :4]
    angles = torch.remainder(params[:, :, 4:8], 2 * torch.pi)
    etas_coef = params[:, :, 8:]
    print(f"xy range: [{xy.min():.2f}, {xy.max():.2f}]")
    print(f"angles range: [{angles.min():.2f}, {angles.max():.2f}]")
    print(f"etas_coef range: [{etas_coef.min():.2f}, {etas_coef.max():.2f}]")
