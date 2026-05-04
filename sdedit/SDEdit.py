"""
Implementation of SDEdit (https://github.com/ermongroup/SDEdit)
Please cite the original paper "SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations" (ICLR 2022)
if you find this algorithm helpful 

In conclusion, SDEdit proposed:
Add noise (perturb) the original image, and then perform reverse diffusion to denoise the perturbed image into editing image
that follows the target prompt. 
"""
from utils.config import BaseConfig
from utils.registry import SDEDIT_REGISTRY
import torch 
from typing import List 
from model import RFModel 
from dataclasses import dataclass
@dataclass
class SDEditConfig(BaseConfig):
    strength: float = 0.95 
    cfg_strength: float = 3.5 
    guidance_scale: float = 3.5 
    num_steps: int = 50 

@SDEDIT_REGISTRY.register('sdedit')
class SDEdit:
    Config = SDEditConfig
    def __init__(self, cfg: SDEditConfig):
        self.cfg = cfg
    
    def perturb(
        self, 
        src: torch.Tensor,
        timesteps: List[torch.Tensor],
        noise: torch.Tensor = None
    ):

        noise = torch.randn_like(src) if noise is None else noise
        if self.cfg.strength == 1:
            return noise 
        t = int((1 - self.cfg.strength) * len(timesteps))
        t = timesteps[t]
        t = t.to(src.device) if isinstance(t, torch.Tensor) else t
        return noise * t + src * (1 - t)
    
    def sample(
        self, 
        model: RFModel,
        perturbed_latents: torch.Tensor = None,
        prompt: str | List[str] = None,
        negative_prompt: str | List[str] = "",
        timesteps: List[torch.Tensor] = None,
        mask: torch.Tensor | List[torch.Tensor]= None, # 1 for editied region, 0 for static region
        inpaint_latents: torch.Tensor | List[torch.Tensor] = None, 
        **kwargs
    ):
        timesteps = model.get_schedule(num_steps=self.cfg.num_steps, **kwargs)[0] if timesteps is None else timesteps
        start = int((1 - self.cfg.strength) * len(timesteps))
        return model.sample(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_steps=self.cfg.num_steps,
            cfg_strength=self.cfg.cfg_strength,
            guidance_scale=self.cfg.guidance_scale,
            latents=perturbed_latents,
            start=start,
            mask=mask,
            inpaint_latents=inpaint_latents,
            **kwargs
        )

    def __call__(
        self,
        src: torch.Tensor | List[torch.Tensor] = None,
        src_latents: torch.Tensor | List[torch.Tensor] = None,
        model: RFModel = None, 
        prompt: str | List[str] = None,
        negative_prompt: str | List[str] = "",
        mask: torch.Tensor | List[torch.Tensor]= None, # 1 for editied region, 0 for static region
        inpaint_latents: torch.Tensor | List[torch.Tensor] = None, 
        **kwargs
    ):
        if isinstance(src, list):
            src = torch.stack(src, dim=0)
        if isinstance(src, torch.Tensor):
            height, width = src.shape[-2:]
            kwargs.update({'height': height, 'width': width})
        if src_latents is None:
            src = model.encode(src)
        else:
            src = src_latents
        timesteps, _ = model.get_schedule(num_steps=self.cfg.num_steps, **kwargs)
        perturbed_latents = self.perturb(src, timesteps)
        return self.sample(
            model=model,
            perturbed_latents=perturbed_latents,
            prompt=prompt,
            negative_prompt=negative_prompt,
            timesteps=timesteps,
            mask=mask,
            inpaint_latents=inpaint_latents,
            **kwargs
        )



