'''
Author: wenjun-VCC
Date: 2024-04-04 20:55:49
LastEditors: wenjun-VCC
LastEditTime: 2024-04-05 00:11:22
FilePath: train.py
Description: __discription:__
Email: wenjun.9707@gmail.com
Copyright (c) 2024 by wenjun/VCC, All Rights Reserved. 
'''
import os
import sys


if sys.platform.startswith('win32'):
    root_path = 'E:/00_Code/VSCode/Python/nerf/nerf'
else:
    root_path = '/mnt/d/code/polygen_wenjun/polygen'
sys.path.append(root_path)


import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from modules.data_module import DataModule
from config.config import NeRFConfig




if __name__ == '__main__':
    
    config = NeRFConfig
    datamodule = DataModule(config)
    datamodule.setup()
    print(len(datamodule.train_data))
    dataset = len(datamodule.train_dataloader())
    
    print('nerf debug...')