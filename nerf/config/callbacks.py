'''
Author: wenjun-VCC
Date: 2024-04-05 08:02:31
LastEditors: wenjun-VCC
LastEditTime: 2024-04-05 08:02:53
FilePath: callbacks.py
Description: __discription:__
Email: wenjun.9707@gmail.com
Copyright (c) 2024 by wenjun/VCC, All Rights Reserved. 
'''
from typing import Any
from pytorch_lightning import Callback, LightningModule, Trainer
import math
import os
import pickle
import numpy as np


class LearningRateWarmupCosineDecayCallback(Callback):
    
    def __init__(
        self,
        init_lr:float=1e-10,
        min_lr:float=1e-6,
        max_lr:float=5e-4,
        warmup_epochs:int=5,
        total_epochs:int=100,
        cos_lr_rate:float=0.9):
        super().__init__()
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lr_cos_epoch = int(total_epochs * cos_lr_rate)

        
        self.warmup_steps = None  # To be calculated later
        self.total_training_steps = None
        self.cos_training_steps = None


    def on_train_start(self, trainer, pl_module):
        # Calculate total steps for warmup based on the number of epochs and steps per epoch
        steps_per_epoch = len(trainer.train_dataloader)
        self.total_training_steps = self.total_epochs * len(trainer.train_dataloader)
        self.warmup_steps = self.warmup_epochs * steps_per_epoch
        self.cos_training_steps = self.lr_cos_epoch * steps_per_epoch + self.warmup_steps

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        
        # Calculate the current step
        current_step = trainer.current_epoch * len(trainer.train_dataloader) + batch_idx

        if current_step <= self.warmup_steps:
            # Linear warmup
            lr = ((self.max_lr - self.init_lr) / self.warmup_steps) * current_step + self.init_lr
        elif current_step <= self.cos_training_steps:
            # Cosine annealing
            progress = (current_step - self.warmup_steps) / (self.cos_training_steps - self.warmup_steps)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.max_lr - self.min_lr) * cosine
        else:
            lr = self.min_lr

        # Set the learning rate to the optimizer
        for pg in trainer.optimizers[0].param_groups:
            pg['lr'] = lr


