import random
import numpy as np
from scipy.ndimage import label, convolve
import cv2
import os
from tqdm import tqdm
from pycocotools.coco import COCO
from utils import get_args, set_seed, create_directory, DataGenerator

class SyntheticRealisticDataGenerator(DataGenerator):
    def __init__(self, args, big=False):
        super().__init__(args)
        self.image_size = np.array(args.big_img_size) if big else np.array(args.img_size)
        self.frgd_path = args.frgd_path
        self.bkgd_path = args.bkgd_path
        self.num_sample = args.num_sample_test

        self.y, self.x = np.meshgrid(np.linspace(0, self.image_size[0]-1, self.image_size[0]), \
                           np.linspace(0, self.image_size[1]-1, self.image_size[1]), indexing='ij')
        self.org_pt = np.array([self.image_size[1]//2, self.image_size[0]//2])

    def is_mask_continuous(self, binary_mask):
        _, num_features = label(binary_mask)
        return num_features == 1

    def get_frgd(self):
        self.frgd_masks = np.zeros((self.num_sample, self.image_size[0], self.image_size[1]), dtype=bool)
        self.frgd_objs = np.zeros((self.num_sample, self.image_size[0], self.image_size[1], 3), dtype=np.float64)
        annFile = f'{self.frgd_path}instances_val2017.json'
        coco=COCO(annFile)
        categories = coco.loadCats(coco.getCatIds())
        catNms = [cat['name'] for cat in categories]
        for i in range(self.num_sample):
            no_valid = True
            while no_valid:
                cat = random.choice(catNms)
                catId = coco.getCatIds(catNms=cat)
                imgIds = coco.getImgIds(catIds=catId)
                if len(imgIds) == 0:
                    continue
                imgId = random.choice(imgIds)
                annIds = coco.getAnnIds(imgId, catIds=catId)
                anns = coco.loadAnns(annIds)
                ann = random.choice(anns)

                area = ann['area']
                if area < 40000:
                    continue
                mask = coco.annToMask(ann)
                if not self.is_mask_continuous(mask):
                    continue
                img = coco.loadImgs(imgId)[0]
                imgNm = img['file_name']
                img_array = cv2.imread(f'{self.frgd_path}val2017/{imgNm}')
                if len(img_array.shape) == 2:
                    continue
                obj = img_array * mask[:,:,None]

                scale = max(self.image_size) / min(mask.shape)
                mask = cv2.resize(mask, (int(round(mask.shape[1]*scale)), int(round(mask.shape[0]*scale))))
                mask = mask[mask.shape[0]//2-self.image_size[0]//2:mask.shape[0]//2-self.image_size[0]//2+self.image_size[0], \
                            mask.shape[1]//2-self.image_size[1]//2:mask.shape[1]//2-self.image_size[1]//2+self.image_size[1],]
                obj = cv2.resize(obj, (int(round(obj.shape[1]*scale)), int(round(obj.shape[0]*scale))))
                obj = obj[obj.shape[0]//2-self.image_size[0]//2:obj.shape[0]//2-self.image_size[0]//2+self.image_size[0], \
                          obj.shape[1]//2-self.image_size[1]//2:obj.shape[1]//2-self.image_size[1]//2+self.image_size[1], :]
                self.frgd_masks[i,:,:] = mask
                self.frgd_objs[i,:,:,:] = obj
                no_valid = False
    
    def get_bkgd(self):
        self.bkgd_img_list = os.listdir(self.bkgd_path)
        self.bkgd_objs = np.zeros((self.num_sample, self.image_size[0], self.image_size[1], 3), dtype=np.float64)
        for i in range(self.num_sample):
            obj = cv2.imread(f'{self.bkgd_path}{self.bkgd_img_list[np.random.randint(len(self.bkgd_img_list))]}')
            scale = max(self.image_size) / min(obj.shape[:2])
            obj = cv2.resize(obj, (int(round(obj.shape[1]*scale)), int(round(obj.shape[0]*scale))))
            obj = obj[obj.shape[0]//2-self.image_size[0]//2:obj.shape[0]//2-self.image_size[0]//2+self.image_size[0], \
                      obj.shape[1]//2-self.image_size[1]//2:obj.shape[1]//2-self.image_size[1]//2+self.image_size[1], :]
            self.bkgd_objs[i,:,:,:] = obj

    def get_depth_norm(self, modi_depth, key_pts):
        return (modi_depth - modi_depth.min()) / (modi_depth.max() - modi_depth.min()) * (key_pts[0] - key_pts[1]) + key_pts[1]
    
    def get_depth_real(self, depth_norm):
        return (self.Z_range[1] - self.Z_range[0]) * depth_norm + self.Z_range[0]

    def render_layer(self, depth_map, depth_key_pts, img_sharp, mask=None):
        if isinstance(mask, np.ndarray):
            mask_blurred = np.zeros((self.n_img,*self.image_size), dtype=np.float64)
        img_blurred = np.zeros((self.n_img,*self.image_size,3), dtype=np.float64)
        diff = depth_key_pts[1] - depth_key_pts[0]
        n_key_pts = len(depth_key_pts)
        for jj, depth in enumerate(depth_key_pts):
            sigmas = self.get_kernel_sigma(depth)
            mask_last = (depth_map <= depth-diff) & (depth_map > depth)
            mask_next = (depth_map <= depth) & (depth_map > depth+diff)
            if jj == 0:
                weight = (depth_map > depth).astype(np.float64) + (depth_map - depth - diff) / (-diff) * mask_next
            elif jj == n_key_pts-1:
                weight = (depth - diff - depth_map) / (-diff) * mask_last + (depth_map <= depth).astype(np.float64)
            else:
                weight = (depth - diff - depth_map) / (-diff) * mask_last + (depth_map - depth - diff) / (-diff) * mask_next
            for ii, sigma in enumerate(sigmas):
                kernel = self.get_blur_kernel(sigma)
                if isinstance(mask, np.ndarray):
                    mask_blurred_temp = convolve(mask.astype(np.float64), kernel, mode='reflect')
                    mask_blurred[ii,:,:] += mask_blurred_temp * weight
                img_blurred_temp = convolve(img_sharp, kernel[:,:,None], mode='reflect')
                img_blurred[ii,:,:,:] += img_blurred_temp * weight[:,:,None]
        if isinstance(mask, np.ndarray):
            return mask_blurred.clip(0,1), img_blurred
        else:
            return img_blurred

    def render_image(self, depth_bkgd, depth_frgd, frgd_mask, bkgd_obj, frgd_obj, n_interval=150):
        depth_bkgd_key_pts = np.linspace(depth_bkgd.max(), depth_bkgd.min(), n_interval+1)
        depth_frgd_key_pts = np.linspace(depth_frgd[frgd_mask].max(), depth_frgd[frgd_mask].min(), n_interval+1)
        img_bkgd_blurred = self.render_layer(depth_bkgd, depth_bkgd_key_pts, bkgd_obj)
        mask_frgd_blurred, img_frgd_blurred = self.render_layer(depth_frgd, depth_frgd_key_pts, frgd_obj, frgd_mask)
        img_clean = img_bkgd_blurred * (1 - mask_frgd_blurred[:,:,:,None]) + img_frgd_blurred
        return img_clean

    def generate_synthetic_image(self, i):
        relative_depth_key_pts = np.flip(np.sort(np.random.rand(4))) # (1 ->) [bg1, bg2, fg1, fg2] (-> 0)
        angles = np.random.rand(2) * 2 * np.pi
        modi_depth = (-np.sin(angles[:,None,None]) * (self.x[None,:,:] - self.org_pt[0]) + \
                      np.cos(angles[:,None,None]) * (self.y[None,:,:] - self.org_pt[1]))
        depth_bkgd_norm = self.get_depth_norm(modi_depth[0,:,:], relative_depth_key_pts[:2])
        depth_frgd_norm = self.get_depth_norm(modi_depth[1,:,:], relative_depth_key_pts[2:])
        depth_norm = depth_bkgd_norm * (1 - self.frgd_masks[i,:,:]) + depth_frgd_norm * self.frgd_masks[i,:,:]
        depth_bkgd = self.get_depth_real(depth_bkgd_norm)
        depth_frgd = self.get_depth_real(depth_frgd_norm)
        depth = self.get_depth_real(depth_norm)
        img_clean = self.render_image(depth_bkgd, depth_frgd, self.frgd_masks[i,:,:], self.bkgd_objs[i,:,:,:], self.frgd_objs[i,:,:,:])
        return img_clean, depth
    
    def generate_synthetic_data(self):
        target_path = self.data_path
        create_directory(target_path)
        create_directory(f'{target_path}/clean')
        create_directory(f'{target_path}/noisy')
        create_directory(f'{target_path}/depth_maps')
        self.images_gt = np.zeros((self.num_sample, self.n_img, *self.image_size, 3), dtype=np.float64)
        self.images_ny = np.zeros(self.images_gt.shape, dtype=np.float64)
        self.depth_maps = np.zeros((self.num_sample, *self.image_size), dtype=np.float64)
        alpha_list = np.random.rand(self.num_sample) * (self.alpha[1] - self.alpha[0]) + self.alpha[0]
        for i in tqdm(range(self.num_sample)):
            img_clean, depth_map = self.generate_synthetic_image(i)
            img_gt = img_clean / 255 * alpha_list[i]
            img_ny = np.random.poisson(img_gt).astype(float) + self.sigma * np.random.randn(*img_gt.shape)
            img_ny = img_ny.clip(0, alpha_list[i]).round()

            for ii in range(self.n_img):
                cv2.imwrite(f'{target_path}/clean/{i}_{ii}.png', (img_gt[ii,:,:,:] / alpha_list[i] * 255).astype(np.uint8))
                cv2.imwrite(f'{target_path}/noisy/{i}_{ii}.png', (img_ny[ii,:,:,:] / alpha_list[i] * 255).astype(np.uint8))
            cv2.imwrite(f'{target_path}/depth_maps/{i}.png', (((depth_map - self.Z_range[0]) / (self.Z_range[1] - self.Z_range[0])) * 255).astype(np.uint8))

            self.images_gt[i,:,:,:,:] = img_gt
            self.images_ny[i,:,:,:,:] = img_ny
            self.depth_maps[i,:,:] = depth_map
        np.save(f'{target_path}/images_gt.npy', self.images_gt)
        np.save(f'{target_path}/images_ny.npy', self.images_ny)
        np.save(f'{target_path}/depth_maps.npy', self.depth_maps)
        np.save(f'{target_path}/alphas.npy', alpha_list)

if __name__ == "__main__":

    BIG = False
    args = get_args('data_gen_test')

    # set_seed(1869)

    generator = SyntheticRealisticDataGenerator(args, big=BIG)
    generator.get_frgd()
    generator.get_bkgd()
    generator.generate_synthetic_data()
