import cv2
import numpy as np
import pdb

class Visualizer:
    def __init__(self, rho_prime, img_size=147, gap_v=20, gap_h=5, scale=10, fontsize_scale=0.35):
        self.rho_prime = rho_prime
        self.img_size = img_size
        self.gap_v = gap_v
        self.gap_h = gap_h
        self.scale = scale
        self.fontsize_scale = fontsize_scale
        self.canvas_blank = self.get_blank_canvas()
        self.colormap_f = self.get_color_map()

    def get_blank_canvas(self):
        gradient_bar = (np.ones((1000, 1), dtype=np.float64) * np.linspace(1, 0, 1000)[:,None] * 0.43 + 0.02) / 0.45
        color_bar = cv2.applyColorMap((gradient_bar*255).clip(0,255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
        color_bar = cv2.resize(color_bar, (self.gap_h*2*self.scale, (self.img_size*2+self.gap_v)*self.scale), interpolation=cv2.INTER_NEAREST)
        canvas_blank = np.ones(((self.img_size*2+self.gap_v*3)*self.scale, (self.img_size*5+self.gap_h*5+40)*self.scale, 3), dtype=np.float64) * 255
        canvas_blank[self.gap_v*self.scale:(self.img_size*2+self.gap_v*2)*self.scale, (self.img_size*5+self.gap_h*5)*self.scale:(self.img_size*5+self.gap_h*7)*self.scale, :] = color_bar
        cv2.putText(canvas_blank, '75', ((self.img_size*5+int(self.gap_h*8))*self.scale,(self.img_size*2+int(self.gap_v*1.9))*self.scale), cv2.FONT_HERSHEY_SIMPLEX, self.fontsize_scale*self.scale, (0,0,0), 1*self.scale)
        cv2.putText(canvas_blank, '118', ((self.img_size*5+int(self.gap_h*7.6))*self.scale,int(self.gap_v*1.6)*self.scale), cv2.FONT_HERSHEY_SIMPLEX, self.fontsize_scale*self.scale, (0,0,0), 1*self.scale)
        cv2.putText(canvas_blank, 'cm', ((self.img_size*5+int(self.gap_h*7.6))*self.scale,int(self.gap_v*0.7)*self.scale), cv2.FONT_HERSHEY_SIMPLEX, self.fontsize_scale*self.scale, (0,0,0), 1*self.scale)
        cv2.putText(canvas_blank, 'Noisy input 1', (0,int(self.gap_v*0.7)*self.scale), cv2.FONT_HERSHEY_SIMPLEX, self.fontsize_scale*self.scale, (0,0,0), 1*self.scale)
        cv2.putText(canvas_blank, 'Noisy input 2', (0,(self.img_size+self.gap_v+int(self.gap_v*0.7))*self.scale), cv2.FONT_HERSHEY_SIMPLEX, self.fontsize_scale*self.scale, (0,0,0), 1*self.scale)
        cv2.putText(canvas_blank, 'Restored colormap 1', ((self.img_size+self.gap_h)*self.scale,int(self.gap_v*0.7)*self.scale), cv2.FONT_HERSHEY_SIMPLEX, self.fontsize_scale*self.scale, (0,0,0), 1*self.scale)
        cv2.putText(canvas_blank, 'Restored colormap 2', ((self.img_size+self.gap_h)*self.scale,(self.img_size+int(self.gap_v*1.7))*self.scale), cv2.FONT_HERSHEY_SIMPLEX, self.fontsize_scale*self.scale, (0,0,0), 1*self.scale)

        cv2.putText(canvas_blank, 'Sharpened colormap', ((int(self.img_size*2)+self.gap_h*2)*self.scale,int(self.gap_v*0.7)*self.scale), cv2.FONT_HERSHEY_SIMPLEX, self.fontsize_scale*self.scale, (0,0,0), 1*self.scale)
        cv2.putText(canvas_blank, 'Refocused colormap *', ((int(self.img_size*2)+self.gap_h*2)*self.scale,(self.img_size+int(self.gap_v*1.7))*self.scale), cv2.FONT_HERSHEY_SIMPLEX, self.fontsize_scale*self.scale, (0,0,0), 1*self.scale)
        cv2.putText(canvas_blank, f'* Refocused with a reference of optical power: {self.rho_prime}', ((int(self.img_size*2)+self.gap_h*2)*self.scale,(self.img_size*2+int(self.gap_v*2.7))*self.scale), cv2.FONT_HERSHEY_SIMPLEX, self.fontsize_scale*self.scale*0.8, (0,0,0), 1*self.scale)

        cv2.putText(canvas_blank, 'Confidence map', ((int(self.img_size*3)+self.gap_h*3)*self.scale,int(self.gap_v*0.7)*self.scale), cv2.FONT_HERSHEY_SIMPLEX, self.fontsize_scale*self.scale, (0,0,0), 1*self.scale)
        cv2.putText(canvas_blank, 'Estimated boundary map', ((int(self.img_size*3)+self.gap_h*3)*self.scale,(self.img_size+int(self.gap_v*1.7))*self.scale), cv2.FONT_HERSHEY_SIMPLEX, self.fontsize_scale*self.scale, (0,0,0), 1*self.scale)
        cv2.putText(canvas_blank, 'Ground truth depth map', ((int(self.img_size*4)+self.gap_h*4)*self.scale,int(self.gap_v*0.7)*self.scale), cv2.FONT_HERSHEY_SIMPLEX, self.fontsize_scale*self.scale, (0,0,0), 1*self.scale)
        cv2.putText(canvas_blank, 'Estimated depth map', ((int(self.img_size*4)+self.gap_h*4)*self.scale,(self.img_size+int(self.gap_v*1.7))*self.scale), cv2.FONT_HERSHEY_SIMPLEX, self.fontsize_scale*self.scale, (0,0,0), 1*self.scale)
        return canvas_blank
    
    def get_color_map(self):
        colormap_f = np.zeros((256, 1, 3), dtype=np.uint8)
        colormap_f[:, 0, 1] = np.arange(256)
        return colormap_f

    def visualize(self, I_1, I_2, C_1, C_2, C_shpd, C_refoc, F, B, Z_gt, Z):
        confidence_map_heatmap = cv2.applyColorMap((F * 255).clip(0,255).astype(np.uint8), self.colormap_f)
        gt_depth_heatmap = cv2.applyColorMap(((Z_gt - 0.73) / 0.45 * 255).clip(0,255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
        depth_map_heatmap = cv2.applyColorMap(((Z - 0.73) / 0.45 * 255).clip(0,255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
        depth_map_heatmap[(depth_map_heatmap[:,:,0]==0) & (depth_map_heatmap[:,:,1]==0) & (depth_map_heatmap[:,:,2]==255)] = np.array([0,0,0])
        
        canvas = self.canvas_blank.copy().astype(np.uint8)
        canvas[self.gap_v*self.scale:(self.img_size+self.gap_v)*self.scale, :self.img_size*self.scale, :] = cv2.resize(I_1*255, (self.img_size*self.scale, self.img_size*self.scale), interpolation=cv2.INTER_NEAREST)
        canvas[(self.img_size+self.gap_v*2)*self.scale:(self.img_size*2+self.gap_v*2)*self.scale, :self.img_size*self.scale, :] = cv2.resize(I_2*255, (self.img_size*self.scale, self.img_size*self.scale), interpolation=cv2.INTER_NEAREST)
        canvas[self.gap_v*self.scale:(self.img_size+self.gap_v)*self.scale, (self.img_size+self.gap_h)*self.scale:(self.img_size*2+self.gap_h)*self.scale, :] = cv2.resize(C_1*255, (self.img_size*self.scale, self.img_size*self.scale), interpolation=cv2.INTER_NEAREST)
        canvas[(self.img_size+self.gap_v*2)*self.scale:(self.img_size*2+self.gap_v*2)*self.scale, (self.img_size+self.gap_h)*self.scale:(self.img_size*2+self.gap_h)*self.scale, :] = cv2.resize(C_2*255, (self.img_size*self.scale, self.img_size*self.scale), interpolation=cv2.INTER_NEAREST)

        canvas[self.gap_v*self.scale:(self.img_size+self.gap_v)*self.scale, (self.img_size*2+self.gap_h*2)*self.scale:(self.img_size*3+self.gap_h*2)*self.scale, :] = cv2.resize(C_shpd*255, (self.img_size*self.scale, self.img_size*self.scale), interpolation=cv2.INTER_NEAREST)
        canvas[(self.img_size+self.gap_v*2)*self.scale:(self.img_size*2+self.gap_v*2)*self.scale, (self.img_size*2+self.gap_h*2)*self.scale:(self.img_size*3+self.gap_h*2)*self.scale, :] = cv2.resize(C_refoc*255, (self.img_size*self.scale, self.img_size*self.scale), interpolation=cv2.INTER_NEAREST)

        canvas[self.gap_v*self.scale:(self.img_size+self.gap_v)*self.scale, (self.img_size*3+self.gap_h*3)*self.scale:(self.img_size*4+self.gap_h*3)*self.scale, :] = cv2.resize(confidence_map_heatmap, (self.img_size*self.scale, self.img_size*self.scale), interpolation=cv2.INTER_NEAREST)
        canvas[(self.img_size+self.gap_v*2)*self.scale:(self.img_size*2+self.gap_v*2)*self.scale, (self.img_size*3+self.gap_h*3)*self.scale:(self.img_size*4+self.gap_h*3)*self.scale, :] = cv2.resize((B*255).clip(0,255), (self.img_size*self.scale, self.img_size*self.scale), interpolation=cv2.INTER_NEAREST)[:,:,None]
        canvas[self.gap_v*self.scale:(self.img_size+self.gap_v)*self.scale, (self.img_size*4+self.gap_h*4)*self.scale:(self.img_size*5+self.gap_h*4)*self.scale, :] = cv2.resize(gt_depth_heatmap, (self.img_size*self.scale, self.img_size*self.scale), interpolation=cv2.INTER_NEAREST)
        canvas[(self.img_size+self.gap_v*2)*self.scale:(self.img_size*2+self.gap_v*2)*self.scale, (self.img_size*4+self.gap_h*4)*self.scale:(self.img_size*5+self.gap_h*4)*self.scale, :] = cv2.resize(depth_map_heatmap, (self.img_size*self.scale, self.img_size*self.scale), interpolation=cv2.INTER_NEAREST)
        return canvas