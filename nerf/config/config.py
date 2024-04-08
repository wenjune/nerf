'''
Author: wenjun-VCC
Date: 2024-04-04 20:53:50
LastEditors: wenjun-VCC
LastEditTime: 2024-04-08 10:17:37
FilePath: config.py
Description: __discription:__
Email: wenjun.9707@gmail.com
Copyright (c) 2024 by wenjun/VCC, All Rights Reserved. 
'''
from dataclasses import dataclass
import sys
import os

from datetime import datetime
current_time = datetime.now().strftime("%Y%m%d%H")


if sys.platform.startswith('win32'):
    ROOT_PATH = 'E:/00_Code/VSCode/Python/nerf/nerf'
else:
    ROOT_PATH = '/mnt/d/code/nerf/nerf'

@dataclass
class NeRFConfig:
    
    # for log record
    wandb_project='nerf-reproduce'
    wandb_name='nerf-synthetic-lego'

    root = os.path.join(ROOT_PATH, 'data/nerf_synthetic/lego')
    
    max_epoch: int=100
    warmup_epoch: int=10
    replica: int=50
    nrays_per_iter: int=1024  # n rays for each images per iter
    half_res: bool=False
    batch_size: int=8  # n images per iter
    nworks: int=0
    learning_rate: float=5e-4
    min_lr: float=5e-6
    
    # test and predict
    H = 800  # image height
    W = 800  # image width
    nrays_per_iter_test :int=6400
    groups: int=H*W//nrays_per_iter_test
    pkl_path: str='E:/00_Code/VSCode/Python/nerf/nerf/results/NeRF_inf_2024040810/test_save'  # pkl folder for image
    image_save_path: str=os.path.join(ROOT_PATH, 'results', f'NeRF_inf_image_{current_time}') 
    nangles: int=120  # number of angles for 360 synthetic views
    inf_save_path: str=os.path.join(ROOT_PATH, 'results', f'NeRF_inf_pkl_{current_time}')
    
    ckpt_path: str='E:/00_Code/VSCode/Python/nerf/nerf/results/epoch=48-val_loss=0.02.ckpt'
    resume: str=None
    
    dims = 256
    depth=8
    is_fourier=True
    pe_dim=10
    view_depend=True
    view_dim=4
    
    sample1: int=64
    sample2: int=128
    
    # ckpt config
    save_top_k: int=3
    save_every_n_epoch: int=10
    ckpt_save_path=os.path.join(ROOT_PATH, 'results', f'NeRF_{current_time}', 'ckpt')
    log_dir=os.path.join(ROOT_PATH, 'results', f'NeRF_{current_time}', 'logs')
    

