'''
Author: wenjun-VCC
Date: 2024-04-04 20:55:49
LastEditors: wenjun-VCC
LastEditTime: 2024-04-07 00:26:22
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
    root_path = '/mnt/d/code/nerf'
sys.path.append(root_path)


import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


# Args
import argparse
parser = argparse.ArgumentParser(description='AutoEncoder')

# log parser
parser.add_argument('--wandb', default=False, help='if use wandb to log', type=bool)
# data parser
parser.add_argument('--batch_size', default=8, help='batch size', type=int)
parser.add_argument('--nworks', default=0, help='num_works for dataloader', type=int)
# strategy parser
parser.add_argument('--accelerator', default='auto', help='you can use cpu for debug', type=str)
parser.add_argument('--devices', default='auto', help='gpu list 0,1,2,3')
parser.add_argument('--strategy', default='auto', help='you can use ddp_find_unused_parameters_true', type=str)

args = parser.parse_args()


from modules.data_module import DataModule
from config.config import NeRFConfig
from modules.nerf import PLNeRF
from config.callbacks import LearningRateWarmupCosineDecayCallback




def nerf(config, model, datamodel):
    
    # callbacks
    checkpoint_callback_top_k = ModelCheckpoint(
        monitor='val_loss',
        dirpath=config.ckpt_save_path,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=config.save_top_k,
        mode='min',
    )

    checkpoint_callback_every_n_epochs = ModelCheckpoint(
        dirpath=config.ckpt_save_path,
        filename='epoch-{epoch}',
        every_n_epochs=config.save_every_n_epoch,
        save_top_k=-1,
    )

    lr_scheduler_callback = LearningRateWarmupCosineDecayCallback(
        max_lr=config.learning_rate,
        min_lr=config.min_lr,
        warmup_epochs=config.warmup_epoch,
        total_epochs=config.max_epoch,
    )
    
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        max_epochs=config.max_epoch,
        logger=WandbLogger(
                name=config.wandb_name,
                project=config.wandb_project,
                dir=config.log_dir
            ) if args.wandb else None,
        log_every_n_steps=5,
        callbacks=[lr_scheduler_callback,
                   checkpoint_callback_top_k,
                   checkpoint_callback_every_n_epochs],
        gradient_clip_val=5.,
        gradient_clip_algorithm='norm',
    )
    
    trainer.fit(
        model=model,
        datamodule=datamodel,
        ckpt_path=config.resume,
    )




if __name__ == '__main__':
    
    config = NeRFConfig
    config.batch_size = args.batch_size
    config.nworks = args.nworks
    
    datamodel = DataModule(config)
    model = PLNeRF(config)
    
    nerf(config=config, datamodel=datamodel, model=model)
    
    print('nerf debug...')