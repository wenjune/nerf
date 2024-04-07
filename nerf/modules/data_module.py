'''
Author: wenjun-VCC
Date: 2024-04-03 11:47:19
LastEditors: wenjun-VCC
LastEditTime: 2024-04-07 21:14:15
FilePath: data_module.py
Description: __discription:__
Email: wenjun.9707@gmail.com
Copyright (c) 2024 by wenjun/VCC, All Rights Reserved. 
'''
import os
import json
import cv2
import imageio
import numpy as np

from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader
import torch
from einops import rearrange, repeat, reduce
import pytorch_lightning as pl



class DatasetProvider:
    
    def __init__(
        self,
        root: str,
        mode: str='train',  # 'train', 'test', 'val'
        half_res: bool=False,
    ) -> None:

        self.root = root
        self.half_res = half_res
        self.mode = mode
        
        # load images and camera poses
        self._setup()
    
    
    def _setup(
        self,
    ):
        
        trans_file = 'transforms_'+self.mode+'.json'
        self.meta = json.load(open(os.path.join(self.root, trans_file), 'r'))
        self.frames = self.meta['frames']
        self.camera_angle_x = self.meta['camera_angle_x']
        self.images = []
        self.poses = []
        for frame in self.frames:
            image_file = os.path.join(self.root, frame['file_path']+'.png')
            image = imageio.imread(image_file)
            
            if self.half_res:
                image = cv2.resize(image, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            
            self.images.append(image)
            self.poses.append(frame['transform_matrix'])
        
        # [nimages, 4, 4]
        self.poses = np.stack(self.poses)
        # [nimages, height, width, 4(B G R A)]
        self.images = (np.stack(self.images) / 255.0).astype(np.float32)
        
        self.height = self.images.shape[1]
        self.width = self.images.shape[2]
        
        self.focal = 0.5*self.width / np.tan(0.5 * self.camera_angle_x)
        alpha = self.images[..., [3]]
        rgb = self.images[..., :3]
        # [R G B] image
        self.images = rgb*alpha + (1 - alpha)
        
        

class NerfDataset(Dataset):
    
    def __init__(
        self,
        root: str,
        nrays_per_iter: int,
        mode: str='train',  # 'train', 'test', 'val'
        half_res: bool=False,
        replicate: int=1,
    ) -> None:
        super(NerfDataset, self).__init__()
        
        provider = DatasetProvider(
            root=root,
            mode=mode,
            half_res=half_res,
        )
        
        self.images = provider.images
        self.poses = provider.poses
        self.focal = provider.focal
        self.width = provider.width
        self.height = provider.height
        self.nrays_per_iter = nrays_per_iter
        self.num_images = len(self.images)
        
        self.precrop_iter = 500
        self.precrop_frac = 0.5
        self.n_iter = 0
        self.replicate = replicate
        self.mode = mode
        
        self._initialize()
        
        
    def _initialize(
        self,
    ):
        
        warange = torch.arange(self.width, dtype=torch.float32)
        harange = torch.arange(self.height, dtype=torch.float32)
        
        # y[800, 800] x[800, 800]
        # y [[0,0,0,...],
        #    [1,1,1,...],
        #    ...       ]]
        # x [[1,2,3,...],
        #    [1,2,3,...],
        #    ...       ]]
        y, x = torch.meshgrid(harange, warange)
        
        # image to camera
        self.transformed_x = (x - self.width * 0.5) / self.focal
        self.transformed_y = (y - self.height * 0.5) / self.focal
        
        self.precrop_index = torch.arange(self.width*self.height).view(self.height, self.width)
        
        dH = int(self.height//2 * self.precrop_frac)
        dW = int(self.width//2 * self.precrop_frac)
        
        self.precrop_index = self.precrop_index[
            self.height//2 - dH : self.height//2 + dH,
            self.width//2 - dW : self.width//2 + dW
        ].reshape(-1)
        
        # [nimages, 4, 4]
        poses = torch.FloatTensor(self.poses)
        all_ray_dirs, all_ray_origins = [], []
        
        for i in range(len(self.images)):
            ray_dirs, ray_origins = make_rays(self.transformed_x, self.transformed_y, poses[i])
            all_ray_dirs.append(ray_dirs)
            all_ray_origins.append(ray_origins)
        
        # [nimages, nrays, 3]
        self.all_ray_dirs = torch.stack(all_ray_dirs)
        self.all_ray_origins = torch.stack(all_ray_origins)
        self.images = torch.FloatTensor(self.images).view(self.num_images, -1, 3)
        
        self.name = 'nerf synthetic dataset'
        
    
    def __len__(self):
        
        return self.num_images * self.replicate
    
    
    def __getitem__(self, index):
        
        index = index % self.num_images
        
        self.n_iter += 1
        
        # [nrays, 3]
        ray_dirs = self.all_ray_dirs[index]
        ray_origins = self.all_ray_origins[index]
        image_pixels = self.images[index]
        
        if self.n_iter < self.precrop_iter:
            
            # [nrays//4, 3] first 500 iters center crop images
            ray_dirs = ray_dirs[self.precrop_index]
            ray_origins = ray_origins[self.precrop_index]
            image_pixels = image_pixels[self.precrop_index]
        
        # random select nrays_per_iter rays for each iteration
        nrays = self.nrays_per_iter
        select_inds = np.random.choice(ray_dirs.shape[0], [nrays], replace=False)
        
        ray_dirs = ray_dirs[select_inds]
        ray_origins = ray_origins[select_inds]
        image_pixels = image_pixels[select_inds]
        
        return ray_dirs, ray_origins, image_pixels



class NerfTestDataset(Dataset):
    
    def __init__(self,
        nangles: int,
        root: str,
        nrays_per_iter: int=3200,
        mode: str='test',  # 'test' or 'predict'
    ) -> None:
        super(NerfTestDataset, self).__init__()
        
        
        provider = DatasetProvider(
            root=root,
            mode='test',
        )
        
        self.images = provider.images
        self.poses = provider.poses
        self.focal = provider.focal
        self.width = provider.width
        self.height = provider.height
        self.nrays_per_iter = nrays_per_iter
        self.num_images = len(self.images)
        self.mode = mode
        self.nangles = nangles
        
        assert self.width*self.height % self.nrays_per_iter == 0, 'nrays_per_iter should be divisible by width*height'
        
        self._init()
        
        if mode == 'test':
            self.all_ray_dirs, self.all_ray_origins, self.images = self.test_init()
            self.total_len = self.all_ray_dirs.shape[0]
        if mode == 'predict':
            self.all_ray_dirs, self.all_ray_origins = self.predict_init()
            self.total_len = self.all_ray_dirs.shape[0]
        
    
    def __len__(self):
        
        return self.total_len
    
    
    def __getitem__(self, index):
        
        ray_dirs = self.all_ray_dirs[index]
        ray_origins = self.all_ray_origins[index]
        
        if self.mode == 'test':
            image_pixels = self.images[index]
        else:
            image_pixels = None
        
        return ray_dirs, ray_origins, image_pixels
    
    
    def _init(self):
        
        warange = torch.arange(self.width, dtype=torch.float32)
        harange = torch.arange(self.height, dtype=torch.float32)
        
        # y[800, 800] x[800, 800]
        # y [[0,0,0,...],
        #    [1,1,1,...],
        #    ...       ]]
        # x [[1,2,3,...],
        #    [1,2,3,...],
        #    ...       ]]
        y, x = torch.meshgrid(harange, warange)
        
        # image to camera
        self.transformed_x = (x - self.width * 0.5) / self.focal
        self.transformed_y = (y - self.height * 0.5) / self.focal
    

    def test_init(
        self,
    ):
        
        # [nimages, 4, 4]
        poses = torch.FloatTensor(self.poses)
        all_ray_dirs, all_ray_origins = [], []
        
        for i in range(len(self.images)):
            ray_dirs, ray_origins = make_rays(self.transformed_x, self.transformed_y, poses[i])
            all_ray_dirs.append(ray_dirs)
            all_ray_origins.append(ray_origins)
        
        # [nimages, nrays, 3]
        all_ray_dirs = rearrange(torch.stack(all_ray_dirs), 'n (r s) d -> (n r) s d', s = self.nrays_per_iter)
        all_ray_origins = rearrange(torch.stack(all_ray_origins), 'n (r s) d -> (n r) s d', s = self.nrays_per_iter)
        images = rearrange(torch.FloatTensor(self.images).view(self.num_images, -1, 3), 'n (r s) d -> (n r) s d', s = self.nrays_per_iter)
        
        return all_ray_dirs, all_ray_origins, images
    
    
    # gen 360 rays
    def predict_init(self):
        
        def trans_t(t):
            return np.array([
                [1,0,0,0],
                [0,1,0,0],
                [0,0,1,t],
                [0,0,0,1],
            ], dtype=np.float32)

        def rot_phi(phi):
            return np.array([
                [1,0,0,0],
                [0,np.cos(phi),-np.sin(phi),0],
                [0,np.sin(phi), np.cos(phi),0],
                [0,0,0,1],
            ], dtype=np.float32)

        def rot_theta(th) : 
            return np.array([
                [np.cos(th),0,-np.sin(th),0],
                [0,1,0,0],
                [np.sin(th),0, np.cos(th),0],
                [0,0,0,1],
            ], dtype=np.float32)

        def pose_spherical(theta, phi, radius):
            c2w = trans_t(radius)
            c2w = rot_phi(phi/180.*np.pi) @ c2w
            c2w = rot_theta(theta/180.*np.pi) @ c2w
            c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
            return c2w

        all_ray_dirs = []
        all_ray_origins = []
        
        for th in np.linspace(-180., 180., self.nangles, endpoint=False):
            pose = torch.FloatTensor(pose_spherical(th, -30., 4.))
            ray_dirs, ray_origins = make_rays(self.transformed_x, self.transformed_y, pose)
            all_ray_dirs.append(ray_dirs)
            all_ray_origins.append(ray_origins)

        all_ray_dirs = rearrange(torch.stack(all_ray_dirs), 'n (r s) d -> (n r) s d', s = self.nrays_per_iter)
        all_ray_origins = rearrange(torch.stack(all_ray_origins), 'n (r s) d -> (n r) s d', s = self.nrays_per_iter)

        return all_ray_dirs, all_ray_origins
        

class DataModule(pl.LightningDataModule):
    
    def __init__(self, config) -> None:
        super().__init__()
        
        self.root = config.root
        self.nrays_per_iter = config.nrays_per_iter
        self.nrays_per_iter_test = config.nrays_per_iter_test
        self.half_res = config.half_res
        self.batch_size = config.batch_size
        self.num_works = config.nworks
        self.replica = config.replica
        self.nangles = config.nangles
        
        
    def setup(self, stage: str = None):
        
        self.train_data = NerfDataset(
            root=self.root,
            nrays_per_iter=self.nrays_per_iter,
            mode='train',
            half_res=self.half_res,
            replicate=self.replica,
        )
        
        self.test_data = NerfTestDataset(
            nangles=self.nangles,
            root=self.root,
            nrays_per_iter=self.nrays_per_iter_test,
            mode='test',
        )
        
        self.val_data = NerfDataset(
            root=self.root,
            nrays_per_iter=self.nrays_per_iter,
            mode='val',
            half_res=self.half_res,
        )
        
        self.predict_data = NerfTestDataset(
            nangles=self.nangles,
            root=self.root,
            nrays_per_iter=self.nrays_per_iter_test,
            mode='predict',
        )
    
    
    def train_dataloader(self):
        
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_works,
            shuffle=True,
        )
        
        
    def val_dataloader(self):
        
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_works//2,
        )
        
    
    def test_dataloader(self):
        
        return DataLoader(
            dataset=self.test_data,
            batch_size=1,
            pin_memory=True,
            num_workers=self.num_works//2,
        )
        
        
    def predict_dataloader(self):
        
        return DataLoader(
            dataset=self.predict_data,
            batch_size=1,
            pin_memory=True,
            num_workers=self.num_works//2,
        )
        


# pixel to world coords
def make_rays(x, y, pose):
        
    # colmap special
    directions = torch.stack([x, -y, -torch.ones_like(x)], dim=-1)
    # rotarion param
    camera_mat = pose[:3,:3]
    ray_dirs = directions.reshape(-1, 3) @ camera_mat.T
    # transfer param
    ray_origins = pose[:3,3].repeat(len(ray_dirs), 1)
        
    return ray_dirs, ray_origins