"""
Implementation of DNAEdit (https://github.com/xiechenxi99/DNAEdit_code)
Please cite the original paper "DNAEdit: Direct Noise Alignment for Text- Guided Rectified Flow Editing" (NeurIPS 2025)
if you find this method helpful 

In conclusion, DNAEdit proposed:
1. DNA (Direct Noise Alignment): A method to directly shift a random noise to the 
    desired structured noise, instead of go through inversion which accumulate approximation error.
2. MVG (Mobile Velocity Guidance)

Note that DNA is not actually inversion method, but since it share the same idea to find 
a structured noise aligning well with the original latent, we regard it as inversion method.
"""

from .inversion import Inversion
from dataclasses import dataclass 
from utils.registry import INVERSION_REGISTRY
from utils.config import BaseConfig
from model import RFModel 
import torch 
from typing import * 

@dataclass
class DNAEditConfig(BaseConfig):
    num_inversion_steps: int = 10
    invert_guidance_scale: float = 1.0 # Guidance distilled model (FLUX.1 Dev) use guidance_scale
    edit_guidance_scale: float = 2.5 # Guidance distilled model (FLUX.1 Dev) use guidance_scale
    invert_cfg_strength: float = 1.0 
    edit_cfg_strength: float = 2.5 # Not guidance distilled model (QwenImage) use cfg_strength for true cfg.
    num_steps: int = 50
    eta: float = 0.85 
    type: str = 'dna-edit'

@INVERSION_REGISTRY.register('dna-edit')
class DNAEdit(Inversion):
    Config = DNAEditConfig
    def __init__(self, config: DNAEditConfig):
        self.cfg = config
    
    def invert(
        self,
        sample: torch.Tensor,
        prompt: str | List[str],
        model: RFModel,
        height: int, 
        width: int,
        **kwargs
    ):
        """
        DNA as inversion, algorithm 1 in the paper
        """
        sample, prompt, bs = self.align_bs(sample, prompt)
        model.set_prompt(prompt)
        # NOTE: A difference between DNAEdit and other inversion method (RF-Inversion)
        # The inversion timesteps of former is a truncated list that does not span over 0 to 1
        # But the timesteps of latter does.
        N = self.cfg.num_steps
        n = self.cfg.num_inversion_steps
        timesteps, _ = model.get_schedule(num_steps=N, height=height, width=width)
        timesteps = torch.flip(timesteps, [0])
        
        latents = []
        velocities = []
        offsets = []

        x_curr = sample.clone()
        noise = torch.randn_like(x_curr)
        for t_curr, t_prev in zip(timesteps[:n+1], timesteps[1:n+2]):
            # Z_t^\star in the paper, calculated by euler integration, with noise - x_curr serving the velocity
            x_prev = (t_prev - t_curr) / (1 - t_curr) * (noise - x_curr) + x_curr

            v_pred = model.get_v_prediction(
                x_prev, 
                t_prev, 
                cfg_strengh=self.cfg.invert_cfg_strength,
                guidance_scale=self.cfg.invert_guidance_scale,
                height=height,
                width=width
            )

            # Equation (6) in the paper. delta v^{DNA}
            delta_v = (x_prev - x_curr) / (t_prev - t_curr) - v_pred

            # Equation (7) in the paper. Latent update. Z_t in the paper, which become Z_{t+1} in next iteration
            dx = delta_v * (t_prev - t_curr)
            x_curr = x_prev - dx 
            # Equation (6) in the paper. Noise update. S_t in the paper.
            noise -= delta_v * (1 - t_curr)

            latents.append(x_curr.cpu())
            velocities.append(v_pred.cpu())
            offsets.append(dx.cpu())

        # Reverse lists for generation (which goes forward in time 1.0 -> 0.0)
        latents = latents[::-1]
        velocities = velocities[::-1]
        offsets = offsets[::-1]

        return {
            "latents": latents,
            "velocities": velocities,
            "offsets": offsets,
            "inverted_latents": x_curr
        }


    def sample(
        self,
        prompt: str | List[str],
        inverted_latents: torch.Tensor,
        src_latents: torch.Tensor, 
        offsets: List[torch.Tensor],
        velocities: List[torch.Tensor],
        model: RFModel, 
        height: int, 
        width: int,
        **kwargs
    ):
        """
        MVG for sampling, algorithm 2 in the paper
        """
        N = self.cfg.num_steps
        n = self.cfg.num_inversion_steps
        model.set_prompt(prompt)
        timesteps, _ = model.get_schedule(num_steps=N, height=height, width=width)
        
        timesteps = timesteps[-(n+1):]

        x_curr = inverted_latents.clone()
        x_mvg = src_latents.clone()

        for (t_curr, t_prev), offset, velocity in zip(zip(timesteps[:-1], timesteps[1:]), offsets, velocities):
            # Equation (8) in the paper. x_star refers to z_t^{\star edit}
            x_star = x_curr + offset.to(model.device)
            v_pred = model.get_v_prediction(
                x_star, 
                t_curr, 
                cfg_strengh=self.cfg.edit_cfg_strength,
                guidance_scale=self.cfg.edit_guidance_scale,
                height=height,
                width=width
            )

            # Equation (9) in the paper. x_mvg_prev refers to Z_{t+1}^{mvg}
            x_mvg = x_mvg + (t_prev - t_curr) * (v_pred - velocity.to(model.device))

            # Equation (10) in the paper.
            v_mvg = (x_curr - x_mvg) / t_curr 
            v_edit = self.cfg.eta * v_pred + (1 - self.cfg.eta) * v_mvg

            # Update x_curr with euler step
            x_curr += v_edit * (t_prev - t_curr)
        
        return x_curr 
    
    @torch.no_grad()
    def __call__(
        self,
        src: torch.Tensor | List[torch.Tensor],
        model: RFModel, 
        src_prompt: str | List[str],
        tgt_prompt: str | List[str]
    ):
        if isinstance(src, list):
            src = torch.stack(src, dim=0)
        height, width = src.shape[-2:]
        src = src.to(model.device)
        latents = model.encode(src.to(model.device))
        inverted_rets = self.invert(
            latents, 
            src_prompt,
            model,
            height, 
            width
        )
        sampled_latents = self.sample(
            prompt=tgt_prompt,
            **inverted_rets,
            src_latents = latents,
            model=model,
            height=height,
            width=width
        )
        return model.decode(sampled_latents, height, width, output_type='pil')