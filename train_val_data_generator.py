import cv2
import numpy as np
from tqdm import tqdm
from scipy.ndimage import generate_binary_structure, binary_dilation, convolve
from utils import get_args, set_seed, create_directory

class SyntheticDataGenerator:
    def __init__(self, args):
        self.image_size = args.img_size
        self.data_path = args.data_path
        self.num_sample_train = args.num_sample_train
        self.num_sample_val = args.num_sample_val
        self.num_shape = args.num_shape
        self.Z_range = args.Z_range
        self.s = args.cam_params['s']
        self.rhos = np.array([args.cam_params['rho_1'],args.cam_params['rho_2']])
        self.Sigma_cam = args.cam_params['sigma_cam']
        self.pixel_pitch = args.cam_params['pixel_pitch']
        self.mag = args.mag
        self.alpha = args.alpha
        self.sigma = args.sigma
        self.R = args.R

        self.dep_vis_low = 1.25 * self.Z_range[0] - 0.25 * self.Z_range[1] # lower - (upper - lower) * 0.25
        self.dep_vis_range = 1.25 * (self.Z_range[1] - self.Z_range[0]) # (upper - lower) + (upper - lower) * 0.25
        self.n_img = len(self.rhos)
        self.struct_crop = generate_binary_structure(2, 2)
        self.half_R = self.R // 2
        self.margin_mask = np.zeros((self.image_size[0], self.image_size[1]), dtype=bool)
        self.margin_mask[self.half_R:-self.half_R,self.half_R:-self.half_R] = 1

        self.struct_dist = generate_binary_structure(2, 1)
        self.dist_blank = - np.ones((self.image_size[0], self.image_size[1]), dtype=np.float64)
        self.dist_blank_patch = - np.ones((self.R, self.R), dtype=np.float64)

        self.sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        self.sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
        self.deri_blank = np.zeros((self.n_img, self.image_size[0], self.image_size[1], 3), dtype=np.float64)

    def get_blur_kernel(self, sigma, order=2):
        k = np.ceil(np.abs(sigma) * 3).astype(np.int64)
        x, y = np.meshgrid(np.linspace(-k, k, k * 2 + 1), np.linspace(-k, k, k * 2 + 1))
        psf = np.exp(- np.power((x**2 + y**2) / (2 * sigma**2), order / 2))
        return psf / np.sum(psf)

    def get_kernel_sigma(self, z):
        return np.abs((1 / z - self.rhos) * self.s + 1) * self.Sigma_cam / self.pixel_pitch / self.mag

    def generate_synthetic_image(self, num_obj):
    
        # Create a black image
        imgs = np.zeros((self.n_img, self.image_size[0], self.image_size[1], 3), dtype=np.float64)
        img_aif = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.float64)
        img_color = np.random.randint(0, [255,255,255], size=3) # B, G, R
        imgs[:, :, :, 0], imgs[:, :, :, 1], imgs[:, :, :, 2] = img_color[0], img_color[1], img_color[2]
        img_aif[:, :, 0], img_aif[:, :, 1], img_aif[:, :, 2] = img_color[0], img_color[1], img_color[2]
        boundary_loc = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.float64)
        image_depth = np.ones(boundary_loc.shape, dtype=np.float64) * self.Z_range[1]
        boundary_depth = np.zeros(boundary_loc.shape, dtype=np.float64)
        mask_blank = np.zeros(boundary_loc.shape, dtype=np.float64)
        # Generate shape indicators
        param_obj = np.random.randint(0, [[3,255,255,255]], size=[num_obj,4]) # shape index (0: circle, 1: rectangle, 2: triangle), B, G, R
        param_obj_z = np.sort(np.random.uniform(self.Z_range[0], self.Z_range[1], size=[num_obj,1]), axis=0) # distance, ascending order
        param_obj_z = param_obj_z[np.argsort(-param_obj_z[:,0]), :]
        param_obj = np.concatenate((param_obj_z, param_obj), axis=1)
        center = np.random.uniform(0, [[self.image_size[1],self.image_size[0]]], size=[num_obj,2]) # x, y
        max_size = np.max(np.array(self.image_size) * 0.8)

        for i, ind in enumerate(param_obj[:,1]):
            mask = mask_blank.copy()
            mask_bndry = mask_blank.copy()
            mask_bndry_depth = mask_blank.copy()
            color = param_obj[i,-3:]
            if ind == 0:
                radius = np.random.randint(0, max_size / 2)
                cv2.circle(mask, (int(center[i,0]), int(center[i,1])), radius, 255, -1)
                cv2.circle(mask_bndry, (int(center[i,0]), int(center[i,1])), radius, 255, 1)
                cv2.circle(mask_bndry_depth, (int(center[i,0]), int(center[i,1])), radius, 255, 1)
            elif ind == 1:
                size_angle = np.random.uniform(0, [max_size,max_size,180], size=3)
                rect = ((center[i,0], center[i,1]), (size_angle[0], size_angle[1]), size_angle[2])
                box = cv2.boxPoints(rect)
                box = np.int64(box)
                cv2.drawContours(mask, [box], 0, 255, -1)
                cv2.drawContours(mask_bndry, [box], 0, 255, 1)
                cv2.drawContours(mask_bndry_depth, [box], 0, 255, 1)
            elif ind == 2:
                size_angle = np.random.uniform(0, [max_size, 2*np.pi, 2*np.pi, 2*np.pi], size=4)
                vertex_x = center[i,0] + size_angle[0] * np.cos(size_angle[1:])
                vertex_y = center[i,1] + size_angle[0] * np.sin(size_angle[1:])
                vertex = np.concatenate((vertex_x.reshape(-1,1), vertex_y.reshape(-1,1)), axis=1).astype(np.int64)
                cv2.drawContours(mask, [vertex], 0, 255, -1)
                cv2.drawContours(mask_bndry, [vertex], 0, 255, 1)
                cv2.drawContours(mask_bndry_depth, [vertex], 0, 255, 1)
            mask_depth_fill = binary_dilation(mask>0, structure=self.struct_crop, iterations=1).astype(np.float64)
            mask_bndry_depth = binary_dilation(mask_bndry_depth>0, structure=self.struct_crop, iterations=1).astype(np.float64)
            mask_img_depth = mask * param_obj[i,0] / 255
            mask_img_depth_ind = np.where(mask_img_depth > 0)
            image_depth[mask_img_depth_ind] = mask_img_depth[mask_img_depth_ind]
            mask_bndry_depth = mask_bndry_depth * param_obj[i,0]
            mask_bndry_depth_ind = np.where(mask_depth_fill > 0)
            boundary_depth[mask_bndry_depth_ind] = mask_bndry_depth[mask_bndry_depth_ind]

            sigmas = self.get_kernel_sigma(param_obj[i,0])
            for ii, sigma in enumerate(sigmas):
                kernel = self.get_blur_kernel(sigma)
                mask_blurred = convolve(mask, kernel, mode='reflect')
                shape_blurred_ind = np.where(mask_blurred > 0)
                for j in range(3):
                    imgs[ii, :, :, j][shape_blurred_ind] = mask_blurred[shape_blurred_ind] / 255 * color[j] + (1 - mask_blurred[shape_blurred_ind] / 255) * imgs[ii, :, :, j][shape_blurred_ind]
            shape_ind = np.where(mask > 0)
            boundary_loc[shape_ind] = mask_bndry[shape_ind]
            for j in range(3):
                img_aif[:, :, j][shape_ind] = mask[shape_ind] / 255 * color[j] + (1 - mask[shape_ind] / 255) * img_aif[:, :, j][shape_ind]
        
        dist = 0
        boundary_dist = self.dist_blank.copy()
        prev_front_mask = boundary_loc>0
        boundary_dist[prev_front_mask] = dist
        if np.count_nonzero(boundary_dist != -1) == 0:
            boundary_dist *= -1
        else:
            while np.count_nonzero(boundary_dist == -1) > 0:
                dist += 1
                curr_front_mask = binary_dilation(prev_front_mask, structure=self.struct_dist, iterations=1)
                boundary_dist[curr_front_mask & np.invert(prev_front_mask)] = dist
                prev_front_mask = curr_front_mask

        imgs = imgs.round()
        deri = self.deri_blank.copy()
        for ii in range(self.n_img):
            deri[ii,:,:,:] = np.sqrt(convolve(imgs[ii,:,:,:], self.sobel_x[:,:,None])**2 + \
                                     convolve(imgs[ii,:,:,:], self.sobel_y[:,:,None])**2)
        return imgs, img_aif, boundary_loc, image_depth, boundary_depth, boundary_dist, deri/255

    def generate_synthetic_data(self, train=True):
        num_sample = self.num_sample_train if train else self.num_sample_val
        partition = 'train' if train else 'val'
        visualization_path = f'{self.data_path}/{partition}'
        create_directory(visualization_path)
        create_directory(f'{visualization_path}/aif')
        create_directory(f'{visualization_path}/clean')
        create_directory(f'{visualization_path}/boundary_locations')
        create_directory(f'{visualization_path}/image_depths')
        create_directory(f'{visualization_path}/boundary_depths')
        create_directory(f'{visualization_path}/boundary_distances')
        create_directory(f'{visualization_path}/derivative_maps')
        self.images = np.zeros((num_sample, self.n_img, self.image_size[0], self.image_size[1], 3), dtype=np.float64)
        self.images_aif = np.zeros((num_sample, self.image_size[0], self.image_size[1], 3), dtype=np.float64)
        self.boundary_locations = np.zeros((num_sample, self.image_size[0], self.image_size[1]), dtype=np.float64)
        self.image_depths = np.zeros(self.boundary_locations.shape, dtype=np.float64)
        self.boundary_depths = np.zeros(self.boundary_locations.shape, dtype=np.float64)
        self.boundary_distances = np.zeros(self.boundary_locations.shape, dtype=np.float64)
        self.derivative_maps = np.zeros(self.images.shape, dtype=np.float64)
        num_obj = np.random.randint(self.num_shape[0], self.num_shape[1], size=num_sample)
        for i, n in tqdm(enumerate(num_obj), total=num_sample):
            imgs, img_aif, boundary_loc, image_depth, boundary_depth, boundary_dist, deri = self.generate_synthetic_image(n)
            self.images[i,:,:,:,:] = imgs
            self.images_aif[i,:,:,:] = img_aif / 255
            self.boundary_locations[i,:,:] = boundary_loc
            self.image_depths[i,:,:] = image_depth
            self.boundary_depths[i,:,:] = boundary_depth
            self.boundary_distances[i,:,:] = boundary_dist
            self.derivative_maps[i,:,:,:,:] = deri
            cv2.imwrite(f'{visualization_path}/aif/{i}.png', img_aif.astype(np.uint8))
            cv2.imwrite(f'{visualization_path}/boundary_locations/{i}.png', boundary_loc.astype(np.uint8))
            cv2.imwrite(f'{visualization_path}/image_depths/{i}.png', (((image_depth - self.dep_vis_low) / self.dep_vis_range) * 255).astype(np.uint8))
            cv2.imwrite(f'{visualization_path}/boundary_depths/{i}.png', (((boundary_depth - self.dep_vis_low) / self.dep_vis_range) * 255).clip(0,255).astype(np.uint8))
            cv2.imwrite(f'{visualization_path}/boundary_distances/{i}.png', (boundary_dist / np.max(boundary_dist) * 255).astype(np.uint8))
            for ii in range(self.n_img):
                cv2.imwrite(f'{visualization_path}/clean/{i}_{ii}.png', imgs[ii,:,:,:].astype(np.uint8))
                if np.max(deri) > 0:
                    cv2.imwrite(f'{visualization_path}/derivative_maps/{i}_{ii}.png', (deri[ii,:,:,:] / np.max(deri[ii,:,:,:]) * 255).astype(np.uint8))
                else:
                    cv2.imwrite(f'{visualization_path}/derivative_maps/{i}_{ii}.png', deri[ii,:,:,:].astype(np.uint8))
        np.save(f'{self.data_path}/images_aif_{partition}.npy', self.images_aif)
        np.save(f'{self.data_path}/boundary_locations_{partition}.npy', self.boundary_locations)
        np.save(f'{self.data_path}/image_depths_{partition}.npy', self.image_depths)
        np.save(f'{self.data_path}/boundary_depths_{partition}.npy', self.boundary_depths)
        np.save(f'{self.data_path}/boundary_distances_{partition}.npy', self.boundary_distances)
        np.save(f'{self.data_path}/derivative_maps_{partition}.npy', self.derivative_maps)
    
    def add_noise(self, train=True):
        num_sample = self.num_sample_train if train else self.num_sample_val
        partition = 'train' if train else 'val'
        visualization_path = f'{self.data_path}/{partition}'
        create_directory(visualization_path, overwrite=False)
        create_directory(f'{visualization_path}/noisy')
        self.images_gt = np.zeros(self.images.shape, dtype=np.float64)
        self.images_ny = np.zeros(self.images.shape, dtype=np.float64)
        self.alpha_list = np.random.rand(num_sample) * (self.alpha[1] - self.alpha[0]) + self.alpha[0]
        for i in tqdm(range(num_sample)):
            for ii in range(self.n_img):
                img = self.images[i,ii,:,:,:]
                img_prime = img / 255 * self.alpha_list[i]
                self.images_gt[i,ii,:,:,:] = img_prime
                img_ny = np.random.poisson(img_prime).astype(float) + self.sigma * np.random.randn(*img_prime.shape)
                img_ny = img_ny.clip(0, self.alpha_list[i]).round()
                self.images_ny[i,ii,:,:,:] = img_ny
                cv2.imwrite(f'{visualization_path}/noisy/{i}_{ii}.png', (img_ny / self.alpha_list[i] * 255).astype(np.uint8))
        np.save(f'{self.data_path}/alphas_{partition}.npy', self.alpha_list)
        np.save(f'{self.data_path}/images_gt_{partition}.npy', self.images_gt)
        np.save(f'{self.data_path}/images_ny_{partition}.npy', self.images_ny)
    
    def crop_patch(self, train=True):
        n_patch = self.num_sample_train * 2 if train else self.num_sample_val * 2
        partition = 'train' if train else 'val'
        visualization_path = f'{self.data_path}/patches/{partition}'
        create_directory(visualization_path)
        create_directory(f'{visualization_path}/aif')
        create_directory(f'{visualization_path}/clean')
        create_directory(f'{visualization_path}/noisy')
        create_directory(f'{visualization_path}/boundary_locations')
        create_directory(f'{visualization_path}/image_depths')
        create_directory(f'{visualization_path}/boundary_depths')
        create_directory(f'{visualization_path}/boundary_distances')
        create_directory(f'{visualization_path}/derivative_maps')
        patches_aif_data = np.zeros((n_patch, self.R, self.R, 3), dtype=np.float64)
        patches_gt_data = np.zeros(patches_aif_data.shape, dtype=np.float64)
        patches_ny_data = np.zeros(patches_aif_data.shape, dtype=np.float64)
        patches_bndry_loc_data = np.zeros((n_patch, self.R, self.R), dtype=np.float64)
        patches_img_dep_data = np.zeros(patches_bndry_loc_data.shape, dtype=np.float64)
        patches_bndry_dep_data = np.zeros(patches_bndry_loc_data.shape, dtype=np.float64)
        patches_bndry_dist_data = np.zeros(patches_bndry_loc_data.shape, dtype=np.float64)
        patches_deri_data = np.zeros(patches_gt_data.shape, dtype=np.float64)
        alpha_data = np.zeros(n_patch, dtype=np.float64)
        n_img = self.images_gt.shape[0]
        dilated_boundary_locations = np.zeros(self.boundary_locations.shape, dtype=np.float64)

        for i in range(n_img):
            dilated_bndry_loc = binary_dilation(self.boundary_locations[i,:,:], structure=self.struct_crop, iterations=self.half_R+1)
            dilated_bndry_loc *= self.margin_mask
            dilated_boundary_locations[i,:,:] = dilated_bndry_loc
        candidate_ind = np.where(dilated_boundary_locations)
        num_candidate = candidate_ind[0].shape[0]
        selected_ind = np.random.choice(num_candidate, n_patch, replace=False)
        img_ind = np.random.randint(0, self.n_img, size=n_patch)

        for i in tqdm(range(n_patch)):
            idx = selected_ind[i]
            n_val = candidate_ind[0][idx]
            i_val = img_ind[i]
            h_val = candidate_ind[1][idx]
            w_val = candidate_ind[2][idx]
            h_s = h_val-self.half_R
            h_e = h_val+self.half_R+1
            w_s = w_val-self.half_R
            w_e = w_val+self.half_R+1
            patches_aif_data[i,:,:,:] = self.images_aif[n_val, h_s:h_e, w_s:w_e, :]
            patches_gt_data[i,:,:,:] = self.images_gt[n_val, i_val, h_s:h_e, w_s:w_e, :]
            patches_ny_data[i,:,:,:] = self.images_ny[n_val, i_val, h_s:h_e, w_s:w_e, :]
            patches_bndry_loc_data[i,:,:] = self.boundary_locations[n_val,h_s:h_e,w_s:w_e]
            patches_img_dep_data[i,:,:] = self.image_depths[n_val,h_s:h_e,w_s:w_e]
            patches_bndry_dep_data[i,:,:] = self.boundary_depths[n_val,h_s:h_e,w_s:w_e]

            dist = 0
            bndry_dist_pat = self.dist_blank_patch.copy()
            prev_front_mask = patches_bndry_loc_data[i,:,:]>0
            bndry_dist_pat[prev_front_mask] = dist
            if np.count_nonzero(bndry_dist_pat != -1) == 0:
                bndry_dist_pat *= -1
            else:
                while np.count_nonzero(bndry_dist_pat == -1) > 0:
                    dist += 1
                    curr_front_mask = binary_dilation(prev_front_mask, structure=self.struct_dist, iterations=1)
                    bndry_dist_pat[curr_front_mask & np.invert(prev_front_mask)] = dist
                    prev_front_mask = curr_front_mask
            patches_bndry_dist_data[i,:,:] = bndry_dist_pat

            patches_deri_data[i,:,:,:] = self.derivative_maps[n_val, i_val, h_s:h_e, w_s:w_e, :]
            alpha_data[i] = self.alpha_list[n_val]

            cv2.imwrite(f'{visualization_path}/aif/{i}.png', (patches_aif_data[i,:,:,:] * 255).astype(np.uint8))
            cv2.imwrite(f'{visualization_path}/clean/{i}.png', (patches_gt_data[i,:,:,:] / alpha_data[i] * 255).astype(np.uint8))
            cv2.imwrite(f'{visualization_path}/noisy/{i}.png', (patches_ny_data[i,:,:,:] / alpha_data[i] * 255).astype(np.uint8))
            cv2.imwrite(f'{visualization_path}/boundary_locations/{i}.png', patches_bndry_loc_data[i,:,:].astype(np.uint8))
            cv2.imwrite(f'{visualization_path}/image_depths/{i}.png', ((patches_img_dep_data[i,:,:] - self.dep_vis_low) / self.dep_vis_range * 255).astype(np.uint8))
            cv2.imwrite(f'{visualization_path}/boundary_depths/{i}.png', ((patches_bndry_dep_data[i,:,:] - self.dep_vis_low) / self.dep_vis_range * 255).clip(0,255).astype(np.uint8))
            cv2.imwrite(f'{visualization_path}/boundary_distances/{i}.png', (patches_bndry_dist_data[i,:,:] / np.max(patches_bndry_dist_data[i,:,:]) * 255).astype(np.uint8))
            if np.max(patches_deri_data[i,:,:,:]) > 0:
                cv2.imwrite(f'{visualization_path}/derivative_maps/{i}.png', (patches_deri_data[i,:,:,:] / np.max(patches_deri_data[i,:,:,:]) * 255).astype(np.uint8))
            else:
                cv2.imwrite(f'{visualization_path}/derivative_maps/{i}.png', patches_deri_data[i,:,:,:].astype(np.uint8))

        np.save(f'{self.data_path}/patches/patches_aif_{partition}.npy', patches_aif_data)
        np.save(f'{self.data_path}/patches/patches_gt_{partition}.npy', patches_gt_data)
        np.save(f'{self.data_path}/patches/patches_ny_{partition}.npy', patches_ny_data)
        np.save(f'{self.data_path}/patches/boundary_locations_{partition}.npy', patches_bndry_loc_data)
        np.save(f'{self.data_path}/patches/image_depths_{partition}.npy', patches_img_dep_data)
        np.save(f'{self.data_path}/patches/boundary_depths_{partition}.npy', patches_bndry_dep_data)
        np.save(f'{self.data_path}/patches/boundary_distances_{partition}.npy', patches_bndry_dist_data)
        np.save(f'{self.data_path}/patches/derivative_maps_{partition}.npy', patches_deri_data)
        np.save(f'{self.data_path}/patches/alphas_{partition}.npy', alpha_data)

if __name__ == "__main__":

    args = get_args('data_gen_train_val')

    set_seed(1869)

    generator = SyntheticDataGenerator(args)
    
    print('Generating synthetic data for training set...')
    generator.generate_synthetic_data(train=True)
    print('- Adding noise...')
    generator.add_noise(train=True)
    print('- Cropping to patches...')
    generator.crop_patch(train=True)

    print('Generating synthetic data for validation set...')
    generator.generate_synthetic_data(train=False)
    print('- Adding noise...')
    generator.add_noise(train=False)
    print('- Cropping to patches...')
    generator.crop_patch(train=False)