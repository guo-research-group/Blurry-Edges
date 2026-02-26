"""
Read and display saved LocalStage and GlobalStage parameters from intermediate results.
Shows the raw numerical values extracted from the Blurry-Edges baseline.
"""

import numpy as np
import os

def display_parameters(image_idx=180):
    """Load and display parameters for a specific image."""
    
    base_dir = 'intermediate_results/raw_parameters'
    local_path = f'{base_dir}/img{image_idx:03d}_params_local.npy'
    global_path = f'{base_dir}/img{image_idx:03d}_params_global.npy'
    
    # Load parameters
    params_local = np.load(local_path)
    params_global = np.load(global_path)
    
    print("=" * 80)
    print(f"PARAMETERS FOR IMAGE {image_idx}")
    print("=" * 80)
    
    # ===== LocalStage Parameters =====
    print("\n" + "=" * 80)
    print("LOCAL STAGE PARAMETERS (CNN Output)")
    print("=" * 80)
    print(f"Shape: {params_local.shape}")
    print(f"  [views=2, height_patches={params_local.shape[1]}, width_patches={params_local.shape[2]}, params=10]")
    
    # Extract dimensions
    n_views = params_local.shape[0]
    h_patches = params_local.shape[1]
    w_patches = params_local.shape[2]
    
    # Show sample patches from different locations
    print(f"\n--- VIEW 0 (Left Camera) ---")
    print(f"Total patches: {h_patches} × {w_patches} = {h_patches * w_patches} patches\n")
    
    # Top-left corner patch
    tl_patch = params_local[0, 0, 0, :]
    print(f"Top-Left Patch [0, 0]:")
    print(f"  XY coordinates:  {tl_patch[0:4]}")
    print(f"  Angles:          {tl_patch[4:8]}")
    print(f"  Eta coeffs:      {tl_patch[8:10]}")
    
    # Center patch
    center_y, center_x = h_patches // 2, w_patches // 2
    center_patch = params_local[0, center_y, center_x, :]
    print(f"\nCenter Patch [{center_y}, {center_x}]:")
    print(f"  XY coordinates:  {center_patch[0:4]}")
    print(f"  Angles:          {center_patch[4:8]}")
    print(f"  Eta coeffs:      {center_patch[8:10]}")
    
    # Bottom-right corner patch
    br_patch = params_local[0, -1, -1, :]
    print(f"\nBottom-Right Patch [{h_patches-1}, {w_patches-1}]:")
    print(f"  XY coordinates:  {br_patch[0:4]}")
    print(f"  Angles:          {br_patch[4:8]}")
    print(f"  Eta coeffs:      {br_patch[8:10]}")
    
    # View 1
    print(f"\n--- VIEW 1 (Right Camera) ---")
    center_patch_v1 = params_local[1, center_y, center_x, :]
    print(f"Center Patch [{center_y}, {center_x}]:")
    print(f"  XY coordinates:  {center_patch_v1[0:4]}")
    print(f"  Angles:          {center_patch_v1[4:8]}")
    print(f"  Eta coeffs:      {center_patch_v1[8:10]}")
    
    # ===== GlobalStage Parameters =====
    print("\n" + "=" * 80)
    print("GLOBAL STAGE PARAMETERS (Transformer Refined)")
    print("=" * 80)
    print(f"Shape: {params_global.shape}")
    print(f"  [batch=1, total_patches={params_global.shape[1]}, params=12]")
    print(f"\nNote: 12 params = 10 params (xy, angles, etas) + 2 auxiliary outputs")
    
    # Show sample global parameters
    n_global_patches = params_global.shape[1]
    print(f"\nTotal patches (both views): {n_global_patches}")
    
    # First patch (from view 0)
    first_global = params_global[0, 0, :]
    print(f"\nFirst Patch (View 0, Position [0,0]):")
    print(f"  XY coordinates:  {first_global[0:4]}")
    print(f"  Angles:          {first_global[4:8]}")
    print(f"  Eta coeffs:      {first_global[8:10]}")
    print(f"  Auxiliary:       {first_global[10:12]}")
    
    # Middle patch
    mid_global = params_global[0, n_global_patches // 2, :]
    print(f"\nMiddle Patch (Patch {n_global_patches // 2}):")
    print(f"  XY coordinates:  {mid_global[0:4]}")
    print(f"  Angles:          {mid_global[4:8]}")
    print(f"  Eta coeffs:      {mid_global[8:10]}")
    print(f"  Auxiliary:       {mid_global[10:12]}")
    
    # ===== Parameter Statistics =====
    print("\n" + "=" * 80)
    print("PARAMETER STATISTICS (View 0, LocalStage)")
    print("=" * 80)
    
    view0_params = params_local[0]  # [h_patches, w_patches, 10]
    
    param_names = ['x1', 'y1', 'x2', 'y2', 'angle1', 'angle2', 'angle3', 'angle4', 'eta1', 'eta2']
    
    print(f"\n{'Parameter':<10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 52)
    for i, name in enumerate(param_names):
        param_values = view0_params[:, :, i].flatten()
        print(f"{name:<10} {param_values.mean():>10.4f} {param_values.std():>10.4f} "
              f"{param_values.min():>10.4f} {param_values.max():>10.4f}")
    
    # ===== Eta to Depth Conversion Example =====
    print("\n" + "=" * 80)
    print("ETA TO DEPTH CONVERSION EXAMPLE")
    print("=" * 80)
    print(f"\nCenter Patch Eta Coefficients: η1={center_patch[8]:.4f}, η2={center_patch[9]:.4f}")
    print(f"\nDepth estimation formula: depth = etas2depth(η1, η2)")
    print(f"  - Uses camera calibration matrix K")
    print(f"  - Applies bilinear interpolation with (η1, η2) as weights")
    print(f"  - See utils/depth_etas.py for implementation")
    
    return params_local, params_global


def compare_multiple_images():
    """Compare parameters across multiple images."""
    print("\n" + "=" * 80)
    print("COMPARING PARAMETERS ACROSS IMAGES")
    print("=" * 80)
    
    base_dir = 'intermediate_results/raw_parameters'
    image_indices = [180, 181, 182, 183, 184]
    
    center_etas = []
    
    for idx in image_indices:
        local_path = f'{base_dir}/img{idx:03d}_params_local.npy'
        params = np.load(local_path)
        
        h_patches, w_patches = params.shape[1], params.shape[2]
        center_y, center_x = h_patches // 2, w_patches // 2
        eta1, eta2 = params[0, center_y, center_x, 8:10]
        center_etas.append((idx, eta1, eta2))
    
    print(f"\n{'Image':<8} {'Eta1':>10} {'Eta2':>10}")
    print("-" * 30)
    for idx, eta1, eta2 in center_etas:
        print(f"{idx:<8} {eta1:>10.4f} {eta2:>10.4f}")
    
    print("\nVariation across images:")
    etas1 = [e[1] for e in center_etas]
    etas2 = [e[2] for e in center_etas]
    print(f"  Eta1 std: {np.std(etas1):.4f}")
    print(f"  Eta2 std: {np.std(etas2):.4f}")


if __name__ == '__main__':
    # Display detailed parameters for first test image
    params_local, params_global = display_parameters(image_idx=180)
    
    # Compare center patch across multiple images
    compare_multiple_images()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Loaded LocalStage parameters (CNN output): 10 params per patch")
    print(f"✓ Loaded GlobalStage parameters (Transformer refined): 12 params per patch")
    print(f"✓ Parameters represent: XY coords (4) + Angles (4) + Etas (2)")
    print(f"✓ Etas are converted to depth using camera calibration")
    print(f"\nSaved arrays contain raw numerical values from baseline model.")
    print("=" * 80)
