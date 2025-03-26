import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import numpy as np

class PostProcessBase(nn.Module, ABC):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.R = args.R
        self.batch_size = args.batch_size
        self.w = args.w
        self.lambda_ridge = (args.alpha_lambda * self.R**2)**2
        yy, xx = torch.meshgrid([torch.linspace(-1.0, 1.0, self.R), \
                                 torch.linspace(-1.0, 1.0, self.R)], indexing='ij')
        self.x, self.y = self.get_xy_mat(xx, yy)
        
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)

    @abstractmethod
    def get_xy_mat(self, xx, yy):
        pass

    def dist4edge(self, x, y, angle):
        return (-torch.sin(angle) * (self.x - x) + torch.cos(angle) * (self.y - y))
    
    def dist4axial(self, x, y, angle):
        return torch.cos(angle) * (self.x - x) + torch.sin(angle) * (self.y - y)

    def itemize_params(self, params):
        x0 = params[:, 0, ...].unsqueeze(1).unsqueeze(1)
        y0 = params[:, 1, ...].unsqueeze(1).unsqueeze(1)
        x1 = params[:, 2, ...].unsqueeze(1).unsqueeze(1)
        y1 = params[:, 3, ...].unsqueeze(1).unsqueeze(1)
        theta1 = params[:, 4, ...].unsqueeze(1).unsqueeze(1)
        phi1 = params[:, 5, ...].unsqueeze(1).unsqueeze(1)
        theta2 = params[:, 6, ...].unsqueeze(1).unsqueeze(1)
        phi2 = params[:, 7, ...].unsqueeze(1).unsqueeze(1)
        return x0, y0, x1, y1, theta1, phi1, theta2, phi2

    def params2dists(self, params):
        x0, y0, x1, y1, theta1, phi1, theta2, phi2 = self.itemize_params(params)

        sgn1 = torch.where(torch.remainder(phi1, 2 * torch.pi) < torch.pi, torch.ones_like(phi1), -torch.ones_like(phi1))
        sgn2 = torch.where(torch.remainder(phi2, 2 * torch.pi) < torch.pi, torch.ones_like(phi2), -torch.ones_like(phi2))

        theta1p = theta1 + phi1
        theta2p = theta2 + phi2

        d11_temp = self.dist4edge(x0, y0, theta1)
        d12_temp = self.dist4edge(x0, y0, theta1p)
        d21_temp = self.dist4edge(x1, y1, theta2)
        d22_temp = self.dist4edge(x1, y1, theta2p)

        sgn11 = torch.where(d11_temp < 0, -torch.ones_like(d11_temp), torch.ones_like(d11_temp))
        sgn12 = torch.where(d12_temp < 0, -torch.ones_like(d12_temp), torch.ones_like(d12_temp))
        sgn21 = torch.where(d21_temp < 0, -torch.ones_like(d21_temp), torch.ones_like(d21_temp))
        sgn22 = torch.where(d22_temp < 0, -torch.ones_like(d22_temp), torch.ones_like(d22_temp))

        axial11 = self.dist4axial(x0, y0, theta1)
        axial12 = self.dist4axial(x0, y0, theta1p)
        axial21 = self.dist4axial(x1, y1, theta2)
        axial22 = self.dist4axial(x1, y1, theta2p)

        d11 = torch.where(axial11 < 0, \
                            torch.sqrt(d11_temp**2 + (axial11*self.w)**2) * sgn11, \
                            d11_temp)
        d12 = torch.where(axial12 < 0, \
                            torch.sqrt(d12_temp**2 + (axial12*self.w)**2) * sgn12, \
                            d12_temp)
        d21 = torch.where(axial21 < 0, \
                            torch.sqrt(d21_temp**2 + (axial21*self.w)**2) * sgn21, \
                            d21_temp)
        d22 = torch.where(axial22 < 0, \
                            torch.sqrt(d22_temp**2 + (axial22*self.w)**2) * sgn22, \
                            d22_temp)

        indicator1 = sgn1 * torch.where((sgn1*d11 > 0) & (sgn1*d12 < 0), 1, -1)
        indicator2 = sgn2 * torch.where((sgn2*d21 >= 0) & (sgn2*d22 <= 0), 1, -1)

        dist1 = torch.min(torch.abs(d11), torch.abs(d12)) * indicator1
        dist2 = torch.min(torch.abs(d21), torch.abs(d22)) * indicator2

        return torch.stack([dist1, dist2], dim=1)
    
    def params2etas(self, params):
        return 10**(torch.erf(params) * 2 - 2)

    def dists2indicators(self, dists, etas):
        hdists = 0.5 * (1.0 + torch.erf(dists / (torch.sqrt(torch.tensor(2)) * etas.unsqueeze(2).unsqueeze(2))))
        return torch.stack([(1.0 - hdists[:, 0, ...]) * (1.0 - hdists[:, 1, ...]),
                            hdists[:, 0, ...] * (1.0 - hdists[:, 1, ...]),
                            hdists[:, 1, ...]], dim=1) # (u0, u1, u2)
    
    def normalized_gaussian(self, x, delta=0.07):
        return torch.exp(- x**2 / delta**2)

    @abstractmethod
    def get_adjA(self, A, A2, trA, trA2):
        pass

    def inverse_3by3(self, A):
        trA = torch.diagonal(A, dim1=-2, dim2=-1).sum(-1)
        A2 = torch.matmul(A, A)
        trA2 = torch.diagonal(A2, dim1=-2, dim2=-1).sum(-1)
        A3 = torch.matmul(A2, A)
        trA3 = torch.diagonal(A3, dim1=-2, dim2=-1).sum(-1)
        detA = (torch.pow(trA, 3) - 3 * trA * trA2 + 2 * trA3) / 6
        adjA = self.get_adjA(A, A2, trA, trA2)
        return adjA / detA.unsqueeze(-1).unsqueeze(-1)
    
    def get_image_derivative(self, img):
        deri = torch.sqrt(F.conv2d(img, self.sobel_x, padding='valid', groups=3)**2 + \
                          F.conv2d(img, self.sobel_y, padding='valid', groups=3)**2 + 1e-8)
        return deri

class PostProcessLocalBase(PostProcessBase):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.ridge = self.lambda_ridge * torch.eye(3, device=self.device).unsqueeze(0)

    def get_xy_mat(self, xx, yy):
        return xx.view(1, self.R, self.R).to(self.device), yy.view(1, self.R, self.R).to(self.device)
    
    def get_adjA(self, A, A2, trA, trA2):
        return A2 - trA.unsqueeze(-1).unsqueeze(-1) * A + ((torch.pow(trA, 2) - trA2) / 2).unsqueeze(-1).unsqueeze(-1) * torch.eye(3, device=A.device).unsqueeze(0)

class PostProcessGlobalBase(PostProcessBase):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.ridge = self.lambda_ridge * torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.stride = args.stride
        self.H = args.img_size[0]
        self.W = args.img_size[1]
        self.H_patches = int(np.floor((self.H - self.R) / self.stride) + 1)
        self.W_patches = int(np.floor((self.W - self.R) / self.stride) + 1)
        self.num_patches = torch.nn.Fold(output_size=[self.H, self.W],
                                         kernel_size=self.R,
                                         stride=self.stride)(torch.ones(1, self.R**2,
                                                                        self.H_patches * self.W_patches,
                                                                        device=self.device)).view(self.H, self.W)

    def get_xy_mat(self, xx, yy):
        return xx.view(1, self.R, self.R, 1, 1).to(self.device), yy.view(1, self.R, self.R, 1, 1).to(self.device)
    
    def get_adjA(self, A, A2, trA, trA2):
        return A2 - trA.unsqueeze(-1).unsqueeze(-1) * A + ((torch.pow(trA, 2) - trA2) / 2).unsqueeze(-1).unsqueeze(-1) * torch.eye(3, device=A.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    def local2global_color(self, patches, pair=True):
        if pair:
            return torch.nn.Fold(output_size=[self.H, self.W], kernel_size=self.R, stride=self.stride)(
                    patches.view(self.batch_size*2, 3*self.R**2, -1)).view(self.batch_size, 2, 3, self.H, self.W) / \
                    self.num_patches.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        else:
            return torch.nn.Fold(output_size=[self.H, self.W], kernel_size=self.R, stride=self.stride)(
                    patches.view(self.batch_size, 3*self.R**2, -1)).view(self.batch_size, 3, self.H, self.W) / \
                    self.num_patches.unsqueeze(0).unsqueeze(0)

    def local2global_bndry(self, bndry_patches):
        return torch.nn.Fold(output_size=[self.H, self.W], kernel_size=self.R, stride=self.stride)(
                bndry_patches.view(self.batch_size, self.R**2, -1)).view(self.batch_size, 1, self.H, self.W) / \
                self.num_patches.unsqueeze(0).unsqueeze(0)

    def local2global_depth(self, depth_map, depth_mask):
        num_depth_patches = torch.nn.Fold(output_size=[self.H, self.W], kernel_size=self.R, stride=self.stride)(
                            (depth_mask.view(self.batch_size,self.R**2,self.H_patches*self.W_patches) > 0).to(torch.float32)).view(self.batch_size, self.H, self.W)
        confidence_map = num_depth_patches / self.num_patches.unsqueeze(0)
        depth_map = torch.nn.Fold(output_size=[self.H, self.W], kernel_size=self.R, stride=self.stride)(
                    depth_map.view(self.batch_size, self.R**2, -1)).view(self.batch_size, self.H, self.W) / \
                    torch.where(num_depth_patches > 0, num_depth_patches, torch.ones_like(num_depth_patches))
        return depth_map, confidence_map
    

    