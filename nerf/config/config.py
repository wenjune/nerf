'''
Author: wenjun-VCC
Date: 2024-04-04 20:53:50
LastEditors: wenjun-VCC
LastEditTime: 2024-04-08 20:41:06
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

    # root path for data
    root = os.path.join(ROOT_PATH, 'data/nerf_synthetic/lego')
    
    # images information
    H = 800  # image height
    W = 800  # image width
    
    # train strategy
    max_epoch: int=100
    warmup_epoch: int=10
    replica: int=50
    nrays_per_iter: int=1024  # n rays for each images per iter
    half_res: bool=False
    batch_size: int=8  # n images per iter
    nworks: int=0
    learning_rate: float=5e-4
    min_lr: float=5e-6
    resume: str=None
    
    # model config
    dims = 256
    depth=8
    is_fourier=True
    pe_dim=10
    view_depend=True
    view_dim=4
    
    # coarse to fine
    sample1: int=64
    sample2: int=128
    
    # test and predict strategy
    nrays_per_iter_test :int=6400  # recommend 6400 or 3200 for 800*800 image
    groups: int=H*W//nrays_per_iter_test  # n groups for one images
    # how many images(angles) you want to generate for 360 synthetic views
    nangles: int=120
    # pkl folder for get image
    pkl_path: str='E:/00_Code/VSCode/Python/nerf/nerf/results/NeRF_inf_pkl_2024040820/synthetic_360_save'
    # image save path in data_process.py
    image_save_path: str=os.path.join(ROOT_PATH, 'results', f'NeRF_inf_image_{current_time}') 
    # where to save the pkl files in test and predict
    inf_save_path: str=os.path.join(ROOT_PATH, 'results', f'NeRF_inf_pkl_{current_time}')
    # ckpt path for test and predict
    ckpt_path: str='E:/00_Code/VSCode/Python/nerf/nerf/results/epoch=92-val_loss=0.02.ckpt'
    
    # ckpt config
    save_top_k: int=3
    save_every_n_epoch: int=10
    ckpt_save_path=os.path.join(ROOT_PATH, 'results', f'NeRF_{current_time}', 'ckpt')
    log_dir=os.path.join(ROOT_PATH, 'results', f'NeRF_{current_time}', 'logs')
    

