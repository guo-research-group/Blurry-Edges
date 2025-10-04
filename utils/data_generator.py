import numpy as np

class DataGenerator:
    def __init__(self, args):
        self.data_path = args.data_path
        self.Z_range = args.Z_range
        self.s = args.cam_params['s']
        self.rhos = np.array([args.cam_params['rho_1'],args.cam_params['rho_2']])
        self.Sigma_cam = args.cam_params['sigma_cam']
        self.pixel_pitch = args.cam_params['pixel_pitch']
        self.mag = args.mag
        self.alpha = args.alpha
        self.sigma = args.sigma
        self.n_img = len(self.rhos)
    
    def get_kernel_sigma(self, z):
        return np.abs((1 / z - self.rhos) * self.s + 1) * self.Sigma_cam / self.pixel_pitch / self.mag

    def get_blur_kernel(self, sigma, order=2):
        sigma = max(sigma, 1e-6)
        k = np.ceil(np.abs(sigma) * 3).astype(np.int64)
        x, y = np.meshgrid(np.linspace(-k, k, k * 2 + 1), np.linspace(-k, k, k * 2 + 1))
        psf = np.exp(- np.power((x**2 + y**2) / (2 * sigma**2), order / 2))
        return psf / np.sum(psf)