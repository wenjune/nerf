'''
                       _oo0oo_
                      o8888888o
                      88" . "88
                      (| -_- |)
                      0\  =  /0
                    ___/`---'\___
                  .' \\|     |// '.
                 / \\|||  :  |||// \
                / _||||| -:- |||||- \
               |   | \\\  - /// |   |
               | \_|  ''\---/''  |_/ |
               \  .-\__  '-'  ___/-. /
             ___'. .'  /--.--\  `. .'___
          ."" '<  `.___\_<|>_/___.' >' "".
         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
         \  \ `_.   \_ __\ /__ _/   .-` /  /
     =====`-.____`.___ \_____/___.-`___.-'=====
                       `=---='


     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

           佛祖保佑     永不宕机     永无BUG

Author: wenjun-VCC
Date: 2024-04-04 20:55:49
LastEditors: wenjun-VCC
LastEditTime: 2024-04-08 17:37:45
FilePath: trainer.py
Description: __discription:__
Email: wenjun.9707@gmail.com
Copyright (c) 2024 by wenjun/VCC, All Rights Reserved. 
'''


import sys

if sys.platform.startswith('win32'):
    root_path = 'E:/00_Code/VSCode/Python/nerf/nerf'
else:
    root_path = '/mnt/d/code/nerf/nerf'
sys.path.append(root_path)


import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


# Args
import argparse
parser = argparse.ArgumentParser(description='AutoEncoder')

# log parser
parser.add_argument('--wandb', default=False, help='if use wandb to log', type=bool)
# test for test data
# predict for generate synthetic 360 degree images
parser.add_argument('--mode', default='train', help='train or test or predict', type=str)

# data parser
parser.add_argument('--batch_size', default=4, help='batch size', type=int)
# if 'test' or 'predict', nworks should be 0
parser.add_argument('--nworks', default=0, help='num_works for dataloader', type=int)

# strategy parser
parser.add_argument('--accelerator', default='auto', help='you can use cpu for debug', type=str)
parser.add_argument('--devices', default='auto', help='gpu list 0,1,2,3')
parser.add_argument('--strategy', default='auto', help='you can use ddp_find_unused_parameters_true', type=str)

args = parser.parse_args()


from modules.data_module import DataModule
from config.config import NeRFConfig
from modules.nerf import PLNeRF
from config.callbacks import LearningRateWarmupCosineDecayCallback, ModelCallbacks




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
    
    inf_callback = ModelCallbacks(
        save_path=config.inf_save_path,
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
                   checkpoint_callback_every_n_epochs,
                   inf_callback],
        gradient_clip_val=5.,
        gradient_clip_algorithm='norm',
    )
    
    if args.mode == 'train':
        trainer.fit(
            model=model,
            datamodule=datamodel,
            ckpt_path=config.resume,
        )
    
    if args.mode == 'test':
        trainer.test(
            model=model,
            datamodule=datamodel,
            ckpt_path=config.ckpt_path,
        )
        
    if args.mode == 'predict':
        trainer.predict(
            model=model,
            datamodule=datamodel,
            ckpt_path=config.ckpt_path,
        )


################
################
# usage:
# first change your root path in trainer.py and config.py

# train: python train.py --wandb True --mode train --batch_size 4 --nworks 24 devices 0,1,2,3
# now just for synthetic data
# you can use wandb to log, change the wandb_congif in config.py
# and change your data path in config.py 'root'
# the ckpt will saved in config.py 'ckpt_save_path'

# test: python train.py --mode test --batch_size 4 --nworks 0 devices 0
# you can use the ckpt from train to test
# change the ckpt_path in config.py
# the test result will save in config.py 'inf_save_path'
# pixels will saved as .pkl files
# change the pkl_path in config.py to get image from .pkl files
# then use data_process.py to get image from .pkl files

# predict: python train.py --mode predict --batch_size 4 --nworks 0 devices 0
# load ckpt the you can get synthetic 360 degree .pkl files
# change the inf_save_path in config.py to save the .pkl files
# then use data_process.py to get image from .pkl files

################
################



if __name__ == '__main__':
    
    config = NeRFConfig
    config.batch_size = args.batch_size
    config.nworks = args.nworks
    
    datamodel = DataModule(config)
    model = PLNeRF(config)
    
    nerf(config=config, datamodel=datamodel, model=model)
    
    print('nerf...')
    
    
    