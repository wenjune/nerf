'''
Author: wenjun-VCC
Date: 2024-04-03 16:45:44
LastEditors: wenjun-VCC
LastEditTime: 2024-04-05 14:28:08
FilePath: nerf.py
Description: __discription:__
Email: wenjun.9707@gmail.com
Copyright (c) 2024 by wenjun/VCC, All Rights Reserved. 
'''
import torch.nn as nn
import torch
from torchtyping import TensorType
from einops import rearrange, repeat, reduce
import pytorch_lightning as pl
# from beartype import beartype
import math
import torch.nn.functional as F


class FourierEmbedder(nn.Module):
    
    def __init__(
        self,
        embed_dim: int,
        fourier: bool,
    ) -> None:
        super(FourierEmbedder, self).__init__()

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
        encoded_x = torch.cat([x, encoded_x], dim=-1)  # [bs, 3 * (2 * embed_dim+1)]
        
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
        view_depend: bool=False,
    ) -> None:
        super(NeRF, self).__init__()
        
        self.fourier = fourier
        self.view = view_depend
        
        self.position_embedder = FourierEmbedder(embed_dim=pe_dim, fourier=fourier)
        self.view_embedder = FourierEmbedder(embed_dim=view_dim, fourier=view_depend)
        
        # position and view embed dim
        self.in_dim = (pe_dim*2+1)*3 if fourier else 3
        self.view_dim = (view_dim*2+1)*3 if view_depend else 4
        
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
        
        sigma = self.sigma_head(mlp_out).relu()
        
        mlp_out = torch.cat([view_embed, mlp_out], dim=-1)
        mlp_out = self.head_layer(mlp_out)
        rgb = self.rgb_proj(mlp_out).sigmoid()
        
        return rgb, sigma
        
        

class PLNeRF(pl.LightningModule):
    
    def __init__(self, config) -> None:
        super().__init__()
        
        self.coarse_sample = config.sample1
        self.fine_sample = config.sample2
        
        self.learning_rate = config.learning_rate
        
        self.coarse_model = NeRF(
            dims=config.dims,
            depth=config.depth,
            pe_dim=config.pe_dim,
            fourier=config.is_fourier,
            view_dim=config.view_dim,
            view_depend=config.view_depend,
        )
        
        self.fine_model = NeRF(
            dims=config.dims,
            depth=config.depth,
            pe_dim=config.pe_dim,
            fourier=config.is_fourier,
            view_dim=config.view_dim,
            view_depend=config.view_depend,
        )
        
    
    def forward(self, batch):
        
        ray_dirs, ray_origins, image_pixels = batch
        
        sample_z_vals = torch.linspace(
            2., 6., self.coarse_sample, device=self.device
        ).view(1, self.coarse_sample)
        
        coarse, fine = self.render_rays(
            ray_dirs=ray_dirs,
            ray_oris=ray_origins,
            sample_z_vals=sample_z_vals,
            fine_sample=self.fine_sample,
        )
        
        return coarse, fine
    
    
    def training_step(self, batch, batch_idx):
        
        ray_dirs, ray_origins, image_pixels = batch
        
        coarse, fine = self(batch)
        
        coarse_rgb, coarse_depth, coarse_accm, coarse_weights = coarse
        fine_rgb, fine_depth, fine_accm, fine_weights = fine
        
        coarse_loss = ((coarse_rgb - image_pixels)**2).mean()
        fine_loss = ((fine_rgb - image_pixels)**2).mean()
        
        psnr = -10. * torch.log(fine_loss.detach()) / torch.log(10.)
        
        train_loss = coarse_loss + fine_loss
        
        self.log("fine_loss", fine_loss, on_step=True, sync_dist=True)
        self.log("coarse_loss", coarse_loss, on_step=True, sync_dist=True)
        self.log("train_loss", train_loss, prog_bar=True, on_step=True, sync_dist=True)
        self.log("psnr", psnr, prog_bar=True, on_step=True, sync_dist=True)
        self.log('learning_rate', self.optimizer.param_groups[0]["lr"], on_step=True)
        
        return train_loss
    
    
    def validation_step(self, batch, batch_idx):
        
        self.eval()
        with torch.no_grad():
            
            ray_dirs, ray_origins, image_pixels = batch
        
            coarse, fine = self(batch)
            
            coarse_rgb, coarse_depth, coarse_accm, coarse_weights = coarse
            fine_rgb, fine_depth, fine_accm, fine_weights = fine
            
            coarse_loss = ((coarse_rgb - image_pixels)**2).mean()
            fine_loss = ((fine_rgb - image_pixels)**2).mean()
            
            psnr = -10. * torch.log(fine_loss.detach()) / torch.log(10.)
            
            val_loss = coarse_loss + fine_loss
            
            self.log("val_loss", val_loss, on_step=True, sync_dist=True)
            self.log("val_psnr", psnr, on_step=True, sync_dist=True)
            
            return val_loss
        
        
    def configure_optimizers(self):
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            
        return {"optimizer":self.optimizer}
    
    
    def render_rays(self, ray_dirs, ray_oris, sample_z_vals, fine_sample, wb=True):
        
        # 
        rays, z_vals = self.sample_rays(ray_dirs=ray_dirs, ray_oris=ray_oris, sample_z_values=sample_z_vals)
        view_dirs = self.sample_viewdirs(ray_dirs=ray_dirs)
        
        sigma, rgb = self.coarse_model(rays, view_dirs)
        sigma = sigma.squeeze(dim=-1)
        coarse_rgb, coarse_depth, coarse_accm, coarse_weights = self.volume_rendering(
            sigma=sigma,
            rgb=rgb,
            z_vals=z_vals,
            ray_dirs=ray_dirs,
            wb=wb
        )
        
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = self.sample_pdf(z_vals_mid, weights=coarse_weights[..., 1:-1], nsamples=fine_sample, det=True)
        
        z_samples = z_samples.detach()
        
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), -1)
        
        rays = ray_oris[..., None, :] + ray_dirs[..., None, :] * z_vals[..., :, None]
        
        fine_sigma, fine_rgb = self.fine_model(rays, view_dirs)
        
        fine_rgb, fine_depth, fine_accm, fine_weights = self.volume_rendering(
            sigma=fine_sigma,
            rgb=fine_rgb,
            z_vals=z_vals,
            ray_dirs=ray_dirs,
            wb=wb,
        )
        
        return (coarse_rgb, coarse_depth, coarse_accm, coarse_weights), (fine_rgb, fine_depth, fine_accm, fine_weights)
        
    
    # generate sample points for each ray
    def sample_rays(self, ray_oris, ray_dirs, sample_z_values):
        
        n_ray = ray_oris.shape[1]
        # [nrays, nsamples]
        sample_z_values = sample_z_values.repeat(n_ray, 1)
        
        # r = o + td
        # [nrays, 3]->[nrays, 1, 3] [nrays, nsamples, 1]  [bs, nrays, 3]
        # TODO
        rays = ray_oris[:, :, None, :] + ray_dirs[:, :, None, :] * sample_z_values[None, :, :, None]
        
        return rays, sample_z_values


    def sample_viewdirs(self, ray_dirs):
        
        return ray_dirs / torch.norm(ray_dirs, dim=-1)


    def volume_rendering(self, sigma, rgb, z_vals, ray_dirs, wb=True):
        
        delta_prefix = z_vals[..., 1:] - z_vals[..., :-1]
        delta_addition = torch.full((z_vals.shape[0], 1), 1e10, device=self.device)
        delta = torch.cat([delta_prefix, delta_addition], dim=-1)

        delta = delta * torch.norm(ray_dirs[..., None, :], dim=-1)
        
        alpha = 1. - torch.exp(-sigma * delta)
        
        exp_term = 1. - alpha
        exp_addition = torch.ones(exp_term.shape[0], 1, device=self.device)
        exp_term = torch.cat([exp_addition, exp_term+1e-8], dim=-1)
        
        transmittance = torch.cumprod(exp_term, dim=-1)[..., :-1]
        
        weight = alpha * transmittance
        
        rgb = torch.sum(weight[..., None] * rgb, dim=-2)
        
        depth = torch.sum(weight * z_vals, dim=-1)
        
        acc_map = torch.sum(weight, -1)
        
        if wb:
            
            rgb = rgb + (1. - acc_map[..., None])
            
        return rgb, depth, acc_map, weight


    def sample_pdf(self, bins, weights, nsamples, det=True):
        
        weights = weights + 1e-6
        
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        
        cdf = torch.cumsum(pdf, -1)
        
        cdf = torch.cat([torch.zeros_like(cdf[..., :1], device=self.device), cdf], dim=-1)
        
        if det:
            u = torch.linspace(0., 1., steps=nsamples, device=self.device)
            u = u.expand(list(cdf.shape[-1] + [nsamples]))
            
        else:
            u = torch.rand(list(cdf.shape[-1] + [nsamples]), device=self.device)
            
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds-1, device=self.device), inds-1)
        above = torch.min((cdf.shape[-1]-1)*torch.ones_like(inds, device=self.device), inds)
        
        inds_g = torch.stack([below, above], dim=-1)
        
        matched_shape = [inds_g.shape[0],inds_g.shape[1],inds_g.shape[2]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
        
        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        
        denom = torch.where(denom<1e-6, torch.ones_like(denom, device=self.device), denom)
        
        t = (u - cdf_g[..., 0] / denom)
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
        
        return samples
        
        
    
