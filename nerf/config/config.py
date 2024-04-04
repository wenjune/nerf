from dataclasses import dataclass


@dataclass
class NeRFConfig:
    
    root = 'E:/00_Code/VSCode/Python/nerf/nerf/data/nerf_synthetic/lego/'
    nrays_per_iter: int=40960
    half_res: bool=False
    batch_size: int=16
    nworks: int=0