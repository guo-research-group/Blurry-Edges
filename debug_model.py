import torch
import numpy as np
from models import LocalStage, GlobalStage

print("="*50)
print("CHECKING LOCAL STAGE")
print("="*50)
model = LocalStage()
state = torch.load('./pretrained_weights/pretrained_local_stage.pth', map_location='cpu')
print(f"Model keys: {len(model.state_dict().keys())}")
print(f"State keys: {len(state.keys())}")
print(f"Keys match: {set(model.state_dict().keys()) == set(state.keys())}")
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

model.load_state_dict(state)
model.eval()

# Test with random input
test_input = torch.randn(1, 3, 21, 21) * 100  # Simulate photon count range
with torch.no_grad():
    output = model(test_input)
    print(f"\nTest input shape: {test_input.shape}")
    print(f"Test input range: [{test_input.min():.2f}, {test_input.max():.2f}]")
    print(f"Test output shape: {output.shape}")
    print(f"Test output range: [{output.min():.2f}, {output.max():.2f}]")
    print(f"Test output stats: mean={output.mean():.4f}, std={output.std():.4f}")

print("\n" + "="*50)
print("CHECKING GLOBAL STAGE")
print("="*50)
global_model = GlobalStage(in_parameter_size=38, out_parameter_size=12, device='cpu')
global_state = torch.load('./pretrained_weights/pretrained_global_stage.pth', map_location='cpu')
print(f"Model keys: {len(global_model.state_dict().keys())}")
print(f"State keys: {len(global_state.keys())}")
print(f"Keys match: {set(global_model.state_dict().keys()) == set(global_state.keys())}")

global_model.load_state_dict(global_state)
global_model.eval()

# Test with random input
test_global_input = torch.randn(1, 10, 38)
with torch.no_grad():
    global_output = global_model(test_global_input)
    print(f"\nTest input shape: {test_global_input.shape}")
    print(f"Test output shape: {global_output.shape}")
    print(f"Test output range: [{global_output.min():.2f}, {global_output.max():.2f}]")

print("\n" + "="*50)
print("CHECKING TEST DATA")
print("="*50)
images = np.load('./data_test/regular/images_ny.npy')
depths = np.load('./data_test/regular/depth_maps.npy')
alphas = np.load('./data_test/regular/alphas.npy')
print(f"Images shape: {images.shape}")
print(f"Images range: [{images.min():.2f}, {images.max():.2f}]")
print(f"Images mean: {images.mean():.2f}")
print(f"Alpha range: [{alphas.min():.2f}, {alphas.max():.2f}]")
print(f"Depth range: [{depths.min():.2f}, {depths.max():.2f}]")
