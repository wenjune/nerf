import os
import sys

if sys.platform.startswith('win32'):
    root_path = 'E:/00_Code/VSCode/Python/nerf/nerf'
else:
    root_path = '/mnt/d/code/nerf/nerf'
sys.path.append(root_path)

import cv2
import numpy as np
import pickle
from einops import rearrange

from config.config import NeRFConfig


def pkl_to_image(folder, groups, savepath, H, W):
    
    all_files = [os.path.join(folder,_) for _ in os.listdir(folder)]
    
    assert len(all_files) % groups == 0, 'The number of files should be divisible by groups'
    
    file_groups = [all_files[i:i+groups] for i in range(0, len(all_files), groups)]
    
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    for idx,files in enumerate(file_groups):
        fine_image = []
        coarse_image = []
        
        for i, file in enumerate(files):
            
            with open(file, 'rb') as f:
                data = pickle.load(f)
                fine_pixels = np.array(data['fine_rgb']*255, dtype=np.uint8)
                coarse_pixels = np.array(data['coarse_rgb']*255, dtype=np.uint8)
                fine_image.append(fine_pixels)
                coarse_image.append(coarse_pixels)
                
        fine_image = np.stack(fine_image, axis=0).reshape(-1, 3)
        coarse_image = np.stack(coarse_image, axis=0).reshape(-1, 3)
        fine_image = rearrange(fine_image, '(h w) c -> h w c', h=H, w=W)
        coarse_image  = rearrange(coarse_image, '(h w) c -> h w c', h=H, w=W)
        fine_image = cv2.cvtColor(fine_image, cv2.COLOR_RGB2BGR)
        coarse_image = cv2.cvtColor(coarse_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(savepath, 'fine_{:04d}.png'.format(idx)), fine_image)
        cv2.imwrite(os.path.join(savepath, 'coarse_{:04d}.png'.format(idx)), coarse_image)
            
    
    
if __name__ == '__main__':
    
    config = NeRFConfig()
    
    pkl_to_image(config.pkl_path, config.groups, config.image_save_path, config.H, config.W)
    


