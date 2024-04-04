'''
Author: wenjun-VCC
Date: 2024-04-03 16:45:44
LastEditors: wenjun-VCC
LastEditTime: 2024-04-04 01:56:49
FilePath: nerf.py
Description: __discription:__
Email: wenjun.9707@gmail.com
Copyright (c) 2024 by wenjun/VCC, All Rights Reserved. 
'''
import torch
import torch.nn as nn

from torchtyping import TensorType




class FourierPE(nn.Module):
    
    def __init__(
        self,
        embed_dim: int,
        fourier: bool,
    ) -> None:
        super(FourierPE, self).__init__()

        self.embed_dim = embed_dim
        self.fourier = fourier
           
        
    def forward(
        self,
        x: TensorType['bs', 3, float]  # [bs, 3(x,y,z)]
    ):
        
        if not self.fourier:
            
            return x
        
        # 生成频率系数
        freqs = 2.0 ** torch.arange(self.embed_dim, dtype=torch.float32, device=x.device)
        # 对每个位置维度应用正余弦编码
        encoded_x = [torch.sin(freqs * x[:, :, None]), torch.cos(freqs * x[:, :, None])]  # [bs, 3, embed_dim]
        # 拼接正余弦编码
        encoded_x = torch.cat(encoded_x, dim=-1)  # [bs, 3, 2 * embed_dim]
        encoded_x = encoded_x.view(x.shape[0], -1)  # [bs, 3 * 2 * embed_dim]
        
        return encoded_x
        
        
        
class MLP(nn.Module):
    
    def __init__(self, dim, depth) -> None:
        super(MLP, self).__init__()
        
        self.dim = dim
        self.depth = depth
        
        self.layers = nn.Sequential(*self._make_layer())
        
    
    def _make_layer(self):
        
        layers = []
        
        for i in range(self.depth):
            
            layers.append(
                nn.Linear(self.dim, self.dim)
            )
            layers.append(
                nn.ReLU()
            )
            
        return layers
    
    
    def forward(self, x):
        
        feature = self.layers(x)
        
        return feature
        


class NeRF(nn.Module):
    
    def __init__(
        self,
        dims: int=256,
        depth: int=8,
        pe_dim: int=8,
        fourier: bool=False,
        view_dim: int=4,
        view: bool=False,
    ) -> None:
        super(NeRF, self).__init__()
        
        self.fourier = fourier
        self.view = view
        
        self.position_embedder = FourierPE(embed_dim=pe_dim, fourier=fourier)
        self.view_embedder = FourierPE(embed_dim=view_dim, fourier=view)
        
        # position and view embed dim
        self.pe_dim = pe_dim*2*3 if fourier else 3
        self.view_dim = view_dim*2*4 if view else 4
        
        self.in_layer = nn.Linear(pe_dim, dims)
        self.middle_layer = nn.Linear(dims+pe_dim, dims)
        
        self.sigma_head = nn.Linear(dims, 1)
        
        self.head_layer = nn.Linear(dims+view_dim, dims//2)
        
        self.rgb_proj = nn.Linear(dims//2, 3)
        
        mlp_depth = depth // 2 - 1
        self.first_mlp = MLP(dim=dims, depth=mlp_depth)
        self.second_mlp = MLP(dim=dims, depth=mlp_depth)
        
    
    def forward(self, x, view_dirs):
        
        view_dirs = view_dirs[:, None].expand(x.shape)
        
        x_embed = self.position_embedder(x)
        view_embed = self.view_embedder(view_dirs)
        
        mlp_out = self.first_mlp(x_embed)
        
        x = torch.cat([x_embed, mlp_out], dim=-1)
        mlp_out = self.middle_layer(x).relu()
        
        mlp_out = self.second_mlp(mlp_out)
        
        sigma = self.sigma_head(mlp_out[:, 3]).relu()
        
        mlp_out = torch.cat([view_embed, mlp_out[:, :3]], dim=-1)
        mlp_out = self.head_layer(mlp_out)
        rgb = self.rgb_proj(mlp_out).sigmoid()
        
        return rgb, sigma
        
        
        
        
        
        
        
        



if __name__ == '__main__':
    
    factor = torch.arange(10, dtype=torch.float32)
    sin_fac = torch.sin(2.**factor)
    
    print('nerf debug')
    
    