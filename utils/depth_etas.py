import torch

class DepthEtas:
    def __init__(self, args, device):
        self.s = args.cam_params['s']
        rho_1 = args.cam_params['rho_1']
        rho_2 = args.cam_params['rho_2']
        sigma_cam = args.cam_params['sigma_cam']
        pixel_pitch = args.cam_params['pixel_pitch']
        self.device = device

        norm_factor = args.R // 2
        self.numerator = 2 * self.s**2 * (rho_2 - rho_1)
        self.denominator_constant = - self.s * (rho_1 - rho_2) * (rho_1 * self.s + rho_2 * self.s - 2)
        self.denominator_factor_root = norm_factor * pixel_pitch * args.mag / sigma_cam
        self.denominator_factor = self.denominator_factor_root**2

        self.intercept = (torch.abs(torch.tensor(self.s * (rho_2 - rho_1))) * sigma_cam / pixel_pitch / args.mag / norm_factor).to(device)

        self.theta_mid = torch.tensor(3 / 4 * torch.pi).to(device)
        self.theta_wng = torch.tensor(1 / 4 * torch.pi).to(device)

    def etas2depth(self, eta1, eta2):
        condition1 = -torch.sin(self.theta_wng) * eta1 + torch.cos(self.theta_wng) * (eta2 - self.intercept)
        condition2 = -torch.sin(self.theta_mid) * (eta1 - self.intercept) + torch.cos(self.theta_mid) * eta2
        condition3 = -torch.sin(self.theta_wng) * (eta1 - self.intercept) + torch.cos(self.theta_wng) * eta2
        eta11 = torch.where(condition1 > 0, (eta1 + eta2 - self.intercept) / 2, \
                              torch.where(condition2 > 0, self.intercept + (eta1 - eta2 - self.intercept) / 2, \
                                          torch.where(condition3 < 0, self.intercept + (eta1 + eta2 - self.intercept) / 2, eta1)))
        eta22 = torch.where(condition1 > 0, self.intercept + (eta1 + eta2 - self.intercept) / 2, \
                              torch.where(condition2 > 0, (eta2 - eta1 + self.intercept) / 2, \
                                          torch.where(condition3 < 0, (eta1 + eta2 - self.intercept) / 2, eta2)))
        z = self.numerator / (self.denominator_factor * (eta11**2 - eta22**2) + self.denominator_constant)
        return z
    
    def depth2sigma(self, depth, rho_prime):
        return torch.abs((1 / depth - rho_prime) * self.s + 1) / self.denominator_factor_root