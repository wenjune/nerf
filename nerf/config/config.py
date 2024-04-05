from dataclasses import dataclass
import sys
import os

from datetime import datetime
current_time = datetime.now().strftime("%Y%m%d%H")


if sys.platform.startswith('win32'):
    ROOT_PATH = 'E:/00_Code/VSCode/Python/nerf/nerf'
else:
    ROOT_PATH = '/mnt/d/code/nerf'

@dataclass
class NeRFConfig:
    
    # for log record
    wandb_project='polygen-reproduce'
    wandb_name='vertex-model'

    root = 'E:/00_Code/VSCode/Python/nerf/nerf/data/nerf_synthetic/lego/'
    
    max_epoch: int=10000
    warmup_epoch: int=int(max_epoch*0.1)
    nrays_per_iter: int=1024  # n rays for each images per iter
    half_res: bool=False
    batch_size: int=16  # n images per iter
    nworks: int=0
    learning_rate: float=2e-4
    min_lr: float=5e-6
    
    ckpt_path: str='D:/Projects/polygen/polygen_pytorch/results/vertex-model/ckpt/epoch=189-train_loss=0.95.ckpt'
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
    
