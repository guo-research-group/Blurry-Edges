"""
Comprehensive Intermediate Results Extraction for Blurry-Edges Baseline

Extracts and visualizes:
1. LocalStage feature maps at all layers
2. Wedge parameter maps (xy centers, angles, etas)
3. GlobalStage refinement effects
4. Color estimation results
5. Depth and confidence maps
6. Boundary detection results

Processes 10 test images and saves detailed visualizations.
"""

import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader
from models import LocalStage, GlobalStage
from utils.args import get_args
from utils.depth_etas import DepthEtas
from utils.postprocessing_loss import PostProcessGlobalBase
import warnings
warnings.filterwarnings('ignore')


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
        alphas_path = os.path.join('./data_test/regular', 'alphas.npy')
        if os.path.exists(alphas_path):
            alphas = np.load(alphas_path)
            if len(alphas.shape) == 1:
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
        
            depth_mask = (self.normalized_gaussian(dists[:,0,...]) > 0.5).to(torch.int32)
            depth_mask_temp = (self.normalized_gaussian(dists[:,1,...]) > 0.5).to(torch.int32) * 2
            depth_mask = torch.where((depth_mask_temp == 2) | (dists[:,1,...] >= 0), depth_mask_temp, depth_mask)
            
            depth_map = torch.where(depth_mask == 1, depth_1.unsqueeze(1).unsqueeze(1), \
                                    torch.where(depth_mask == 2, depth_2.unsqueeze(1).unsqueeze(1), depth_mask))

            dists_B = torch.where(dists[:,1,...] >= 0, dists[:,1,...], \
                                  torch.where(torch.abs(dists[:,0,...])<torch.abs(dists[:,1,...]), torch.abs(dists[:,0,...]), torch.abs(dists[:,1,...])))
            local_boundaries = self.normalized_gaussian(dists_B)

        if colors_only:
            return colors
        else:
            return patches, local_boundaries.unsqueeze(1), depth_map, depth_mask

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
            patches, local_boundaries, depth_map, depth_mask = self.get_patches(xy_angles, etas, colors_only)
            global_image = self.local2global_color(patches)
            global_bndry = self.local2global_bndry(local_boundaries)
            global_depth_map, confidence_map = self.local2global_depth(depth_map, depth_mask)
            return global_image.detach().cpu().numpy(), global_bndry.detach().cpu().numpy(), global_depth_map.detach().cpu().numpy(), confidence_map.detach().cpu().numpy()


class LocalStageWithIntermediates(LocalStage):
    """Modified LocalStage that returns intermediate feature maps"""
    
    def forward(self, x):
        intermediates = {}
        
        # Initial conv
        x = self.conv1(x)
        intermediates['conv1'] = x.detach().cpu()
        
        x = self.maxpool1(x)
        intermediates['pool1'] = x.detach().cpu()
        
        # Layer 0
        x = self.layer0(x)
        intermediates['layer0'] = x.detach().cpu()
        
        x = self.maxpool1(x)
        intermediates['pool2'] = x.detach().cpu()
        
        # Layer 1
        x = self.layer1(x)
        intermediates['layer1'] = x.detach().cpu()
        
        # Layer 2
        x = self.layer2(x)
        intermediates['layer2'] = x.detach().cpu()
        
        # Layer 3
        x = self.layer3(x)
        intermediates['layer3'] = x.detach().cpu()
        
        x = self.maxpool2(x)
        intermediates['pool3'] = x.detach().cpu()
        
        # FC
        output = self.fc(x)
        
        return output, intermediates


def visualize_feature_maps(features, title, save_path):
    """Visualize first 16 channels of a feature map"""
    B, C, H, W = features.shape
    n_show = min(16, C)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for i in range(n_show):
        ax = axes[i // 4, i % 4]
        feat = features[0, i, :, :].numpy()
        im = ax.imshow(feat, cmap='viridis')
        ax.set_title(f'Ch {i}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Hide empty subplots
    for i in range(n_show, 16):
        axes[i // 4, i % 4].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_parameter_maps(params_local, params_global, save_path):
    """Visualize spatial distribution of wedge parameters"""
    # params shape: [2, H_patches, W_patches, 10]
    
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 5, figure=fig, hspace=0.3, wspace=0.3)
    
    titles_local = ['XY Center X (View1)', 'XY Center Y (View1)', 'Angle 1 (View1)', 'Angle 2 (View1)', 'Eta Coef']
    titles_global = ['XY Center X (Refined)', 'XY Center Y (Refined)', 'Angle 1 (Refined)', 'Angle 2 (Refined)', 'Eta Coef (Refined)']
    
    indices = [0, 1, 4, 5, 8]
    
    for i, idx in enumerate(indices):
        # Local stage
        ax = fig.add_subplot(gs[0, i])
        data = params_local[0, :, :, idx].cpu().numpy()
        im = ax.imshow(data, cmap='jet', aspect='auto')
        ax.set_title(f'LOCAL: {titles_local[i]}', fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Global stage
        ax = fig.add_subplot(gs[1, i])
        data = params_global[0, :, :, idx].cpu().numpy()
        im = ax.imshow(data, cmap='jet', aspect='auto')
        ax.set_title(f'GLOBAL: {titles_global[i]}', fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_complete_pipeline(img_ny, depth_est, confidence, boundary, gt_depth, save_path):
    """Visualize complete pipeline results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Input views and estimated output
    axes[0, 0].imshow(img_ny[0, 0])
    axes[0, 0].set_title('Input View 1', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_ny[0, 1])
    axes[0, 1].set_title('Input View 2', fontweight='bold')
    axes[0, 1].axis('off')
    
    im = axes[0, 2].imshow(depth_est[0], cmap='plasma')
    axes[0, 2].set_title('Estimated Depth', fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
    
    # Row 2: Confidence, boundary, ground truth
    im = axes[1, 0].imshow(confidence[0], cmap='hot')
    axes[1, 0].set_title('Confidence Map', fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    
    im = axes[1, 1].imshow(boundary[0, 0], cmap='gray')
    axes[1, 1].set_title('Boundary Detection', fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    
    im = axes[1, 2].imshow(gt_depth, cmap='plasma')
    axes[1, 2].set_title('Ground Truth Depth', fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def extract_all_intermediates(args, local_module, global_module, helper, datasetloader, device, num_images=10):
    """Extract all intermediate results"""
    
    output_dir = './intermediate_results'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/feature_maps', exist_ok=True)
    os.makedirs(f'{output_dir}/parameter_maps', exist_ok=True)
    os.makedirs(f'{output_dir}/pipeline_results', exist_ok=True)
    
    print("="*80)
    print("EXTRACTING COMPREHENSIVE INTERMEDIATE RESULTS")
    print("="*80)
    print(f"Processing {num_images} test images from index 180")
    print(f"Output directory: {output_dir}")
    print()
    print("Loading models... ", end='', flush=True)
    print("DONE")
    print("Starting extraction...")
    print()
    
    with torch.no_grad():
        processed = 0
        for j, (img_ny, gt_depth) in enumerate(datasetloader):
            if j < 180:
                continue
            if processed >= num_images:
                break
            
            print(f"\n{'='*80}")
            print(f"Image #{j} (Progress: {processed+1}/{num_images})")
            print(f"{'='*80}")
            processed += 1
            
            # Move inputs to device first
            img_ny = img_ny.to(device)
            gt_depth = gt_depth.to(device)
            
            # Prepare patches
            t_img = img_ny.flatten(0,1).permute(0,3,1,2)
            img_patches = nn.Unfold(args.R, stride=args.stride)(t_img).view(2, 3, args.R, args.R, helper.H_patches, helper.W_patches)
            vec = img_patches.permute(0,4,5,1,2,3).reshape(2 * helper.H_patches * helper.W_patches, 3, args.R, args.R)
            
            # LOCAL STAGE with intermediates
            print("  → Extracting LocalStage features...")
            # Ensure input is on correct device with correct dtype
            vec = vec.to(device=device, dtype=torch.float32)
            params_est, intermediates = local_module(vec)
            
            # Save feature maps for each layer
            for layer_name, features in intermediates.items():
                if 'pool' not in layer_name:  # Skip pooling layers for clarity
                    save_path = f'{output_dir}/feature_maps/img{j:03d}_{layer_name}.png'
                    visualize_feature_maps(features, f'Image {j} - {layer_name.upper()} Features', save_path)
                    print(f"     ✓ Saved {layer_name} features")
            
            # LOCAL STAGE parameters
            params_local = params_est.view(2, helper.H_patches, helper.W_patches, 10).detach()
            xy = params_local[:, :, :, :4]
            angles = torch.remainder(params_local[:, :, :, 4:8], 2 * torch.pi)
            etas_coef = params_local[:, :, :, 8:]
            params_local_formatted = torch.cat([xy, angles, etas_coef], dim=3)
            
            # Prepare for global stage
            colors = helper(params_est.view(2, helper.H_patches, helper.W_patches, 10).flatten(start_dim=1,end_dim=2), 
                          img_patches, colors_only=True).flatten(start_dim=3,end_dim=4).flatten(start_dim=1,end_dim=2).permute(0,2,1)
            # Ensure all tensors are on same device
            xy_device = xy.view(2, -1, 4).to(device)
            angles_device = angles.view(2, -1, 4).to(device)
            etas_device = etas_coef.view(2, -1, 2).to(device)
            colors_device = colors.to(device)
            pm = torch.cat([xy_device / 3, \
                          (angles_device - torch.pi) / torch.pi, \
                          etas_device - 0.5, \
                          (colors_device - 0.5) * 2], dim=2).unsqueeze(0).permute(0,2,1,3).flatten(2,3)
            
            # GLOBAL STAGE refinement
            print("  → Processing GlobalStage refinement...")
            pm = pm.to(device)  # Ensure on correct device
            params_refined = global_module(pm)
            xy_refined = params_refined[:, :, :4] * 3
            angles_refined = torch.remainder((params_refined[:, :, 4:8] + 1) * torch.pi, 2 * torch.pi)
            etas_refined = params_refined[:, :, 8:] + 0.5
            est = torch.cat([xy_refined, angles_refined, etas_refined], dim=2)
            
            # Reshape for visualization
            # est shape: [1, H_patches*W_patches, 12]
            num_patches = helper.H_patches * helper.W_patches
            params_global = est.view(1, helper.H_patches, helper.W_patches, 12)
            
            # Extract relevant parameters (similar to local stage format)
            params_global_formatted = torch.cat([
                params_global[:, :, :, 0:4],   # xy coordinates
                params_global[:, :, :, 4:8],   # angles
                params_global[:, :, :, 8:12]   # etas
            ], dim=3)
            
            # Save RAW parameter arrays (numerical values)
            print("  → Saving raw parameter arrays...")
            os.makedirs(f'{output_dir}/raw_parameters', exist_ok=True)
            
            # LocalStage parameters: [2, H_patches, W_patches, 10]
            # Dimension breakdown:
            #   [0:4]  = xy coordinates (wedge centers for 2 views)
            #   [4:8]  = angles (wedge orientations for 2 views)
            #   [8:10] = eta coefficients (depth-related parameters)
            np.save(f'{output_dir}/raw_parameters/img{j:03d}_params_local.npy', 
                    params_local.cpu().numpy())
            
            # GlobalStage parameters: [1, H_patches*W_patches, 12]
            # Refined parameters after global refinement
            np.save(f'{output_dir}/raw_parameters/img{j:03d}_params_global.npy', 
                    params_global.cpu().numpy())
            
            print(f"     Saved raw parameter arrays")
            
            # Save parameter maps (visualization)
            print("  Visualizing parameter maps...")
            save_path = f'{output_dir}/parameter_maps/img{j:03d}_params.png'
            visualize_parameter_maps(params_local_formatted, params_global_formatted, save_path)
            print(f"     Saved parameter comparison")
            
            # COMPLETE PIPELINE
            print("  Running complete pipeline...")
            col_est, bndry_est, depth_map, confidence_map = helper(est, img_patches, colors_only=False)
            
            # Save complete results
            save_path = f'{output_dir}/pipeline_results/img{j:03d}_complete.png'
            visualize_complete_pipeline(img_ny.cpu().numpy(), depth_map, confidence_map, 
                                       bndry_est, gt_depth[0].cpu().numpy(), save_path)
            print(f"     Saved complete pipeline visualization")
            
            # Print statistics
            coverage = (confidence_map > 0.05).sum() / confidence_map.size * 100
            print(f"\n  Statistics:")
            print(f"    Depth coverage: {coverage:.1f}%")
            print(f"    Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
            print(f"    Confidence range: [{confidence_map.min():.3f}, {confidence_map.max():.3f}]")
    
    print("\n" + "="*80)
    print("✓ EXTRACTION COMPLETE!")
    print("="*80)
    print(f"\nSaved results to: {output_dir}/")
    print("\nGenerated files:")
    print(f"  1. Feature maps: {output_dir}/feature_maps/")
    print(f"     - conv1, layer0, layer1, layer2, layer3 features for each image")
    print(f"  2. Parameter maps: {output_dir}/parameter_maps/")
    print(f"     - Spatial distribution of wedge parameters (local vs global)")
    print(f"  3. Pipeline results: {output_dir}/pipeline_results/")
    print(f"     - Complete pipeline: inputs → depth → confidence → boundaries")
    print(f"  4. Raw parameters: {output_dir}/raw_parameters/")
    print(f"     - Numerical arrays of LocalStage and GlobalStage parameters")
    print("="*80)
    
    # Create README explaining the parameters
    readme_path = f'{output_dir}/PARAMETERS_GUIDE.txt'
    with open(readme_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BLURRY-EDGES PARAMETERS GUIDE\n")
        f.write("="*80 + "\n\n")
        f.write("WHAT ARE THESE PARAMETERS?\n")
        f.write("-"*80 + "\n")
        f.write("Blurry-Edges uses coded aperture imaging to estimate depth from blur patterns.\n")
        f.write("The LocalStage CNN extracts 10 parameters per image patch describing the blur.\n\n")
        f.write("PARAMETER BREAKDOWN (LocalStage Output):\n")
        f.write("-"*80 + "\n")
        f.write("File: raw_parameters/imgXXX_params_local.npy\n")
        f.write("Shape: [2, H_patches, W_patches, 10]\n\n")
        f.write("Dimensions:\n")
        f.write("  [0] = View 1 (first image in stereo pair)\n")
        f.write("  [1] = View 2 (second image in stereo pair)\n")
        f.write("  [2,3] = Spatial location (row, column of patch in image grid)\n")
        f.write("  [4] = 10 parameters per patch:\n\n")
        f.write("Parameter indices [0:10]:\n")
        f.write("  [0:4]  - XY Coordinates: (x1, y1, x2, y2)\n")
        f.write("           Location of two wedge centers in the blur pattern\n\n")
        f.write("  [4:8]  - Angles: (θ1, θ2, θ3, θ4)\n")
        f.write("           Orientation of blur wedges in radians [0, 2π]\n\n")
        f.write("  [8:10] - Eta Coefficients: (η1, η2)\n")
        f.write("           Depth-related blur spread parameters\n")
        f.write("           Used to compute actual depth via: depth = etas2depth(η1, η2)\n\n")
        f.write("GLOBAL STAGE REFINEMENT:\n")
        f.write("-"*80 + "\n")
        f.write("File: raw_parameters/imgXXX_params_global.npy\n")
        f.write("Shape: [1, N_patches, 12]\n\n")
        f.write("The GlobalStage transformer refines these parameters using spatial context.\n")
        f.write("12 parameters = 4 xy + 4 angles + 4 etas (refined versions)\n\n")
        f.write("HOW TO LOAD AND USE:\n")
        f.write("-"*80 + "\n")
        f.write("import numpy as np\n\n")
        f.write("# Load local parameters\n")
        f.write("params = np.load('raw_parameters/img180_params_local.npy')\n")
        f.write("print(f'Shape: {params.shape}')  # [2, H, W, 10]\n\n")
        f.write("# Get parameters for specific patch (view=0, row=5, col=3)\n")
        f.write("patch_params = params[0, 5, 3, :]\n\n")
        f.write("# Extract components\n")
        f.write("xy_centers = patch_params[0:4]    # Wedge centers\n")
        f.write("angles = patch_params[4:8]        # Orientations\n")
        f.write("etas = patch_params[8:10]         # Depth coefficients\n\n")
        f.write("print(f'XY: {xy_centers}')\n")
        f.write("print(f'Angles: {angles}')\n")
        f.write("print(f'Etas: {etas}')\n\n")
        f.write("="*80 + "\n")
    
    print(f"\n✓ Created parameter guide: {readme_path}\n")


if __name__ == "__main__":
    args = get_args('eval')
    device = torch.device(args.cuda)
    
    # Load dataset
    dataset_test = TestDataset(device, data_path=args.data_path)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)
    
    # Load models with intermediate extraction
    local_module = LocalStageWithIntermediates().to(device)
    local_module.load_state_dict(torch.load(f'{args.model_path}/pretrained_local_stage.pth', map_location=device))
    local_module.eval()
    
    global_module = GlobalStage(in_parameter_size=38, out_parameter_size=12, device=device).to(device)
    if args.densify == 'w':
        global_module.load_state_dict(torch.load(f'{args.model_path}/pretrained_global_stage_w.pth', map_location=device))
    else:
        global_module.load_state_dict(torch.load(f'{args.model_path}/pretrained_global_stage.pth', map_location=device))
    global_module.eval()
    
    # Helper
    depthCal = DepthEtas(args, device)
    helper = PostProcess(args, depthCal, device)
    
    # Extract intermediates
    extract_all_intermediates(args, local_module, global_module, helper, test_loader, device, num_images=10)
