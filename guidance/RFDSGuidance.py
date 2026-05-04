"""
Implementation of RFDS (https://github.com/yangxiaofeng/rectified_flow_prior)
Please cite the original paper "Text-to-Image Rectified Flow as Plug-and-Play Priors" (ICLR 2025)
if you find this algorithm useful 
"""
from model import RFModel 
from torch import nn 
import torch 
from torch.nn import functional as F
from utils.registry import GUIDANCE_REGISTRY
from typing import * 
from utils.config import BaseConfig
from dataclasses import dataclass
@dataclass
class RFDSGuidanceConfig(BaseConfig):
    device: str = 'cuda'

@GUIDANCE_REGISTRY.register('rfds')
class RFDSGuidance(nn.Module):
    def __init__(
        self,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device 

    def forward(self, model: RFModel, x: torch.Tensor, t: torch.Tensor, cond: Dict, cfg_strength: int, noise=None):
        latents = model.encode(x)
        height, width = x.shape[-2:]
        if noise is None:
            noise = torch.randn_like(latents)
        latents_noisy = t.view(-1, 1, 1) * noise + (1 - t.view(-1, 1, 1)) * latents

        with torch.no_grad():
            v_pred = model.get_v_prediction(
                latents_noisy, 
                t, 
                cond=cond,
                cfg_strength=cfg_strength,
                width=width,
                height=height,
                norm_cfg = False
            )

        target = torch.nan_to_num(v_pred)
        target = (target).detach() 
        loss_rfds = F.mse_loss(noise - latents, target, reduction="sum") / latents.shape[0]
        
        return {
            "loss": loss_rfds.mean(),
            "velocity": v_pred.detach().cpu(),
            "latents": latents.detach().cpu()
            }

