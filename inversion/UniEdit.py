"""
Implementation of UniEdit-Flow (https://github.com/DSL-Lab/UniEdit-Flow)
Please cite the original paper "UniEdit-Flow: Unleashing Inversion and Editing in the Era of Flow Models" (ICLR 2026)
if you find this method helpful.

In conclusion, UniEdit proposed
* Uni-Inv: Evaluate velocity at t_prev (t_i) instead of at t_curr (t_{i+1})
  This is somehow similar to the FireFlow, which approximate velocity at midpoint instead
* Uni-Edit: 
"""

from .inversion import Inversion
from dataclasses import dataclass 
from utils.registry import INVERSION_REGISTRY
from utils.config import BaseConfig
from model import RFModel 
import torch 
from typing import * 
from tqdm import tqdm 

@dataclass
class UniEditConfig(BaseConfig):
    invert_guidance_scale: float = 1.0
    edit_guidance_scale: float = 4.0 
    invert_cfg_strength: float = 1.0 
    edit_cfg_strength: float = 3.5 
    num_steps: int = 28 
    alpha: float = 0.6 
    omega: float = 5.0 
    zero_init: bool = False 
    type: str = 'uniedit'

@INVERSION_REGISTRY.register('uniedit')
class UniEdit(Inversion):
    Config = UniEditConfig

    def __init__(self, cfg: UniEditConfig):
        self.cfg = cfg 
    
    def invert(
        self,
        sample, 
        prompt, 
        model,
        height, 
        width,
        verbose = True,
        return_trajectory: bool = False,
        **kwargs
    ):
        """
        Algorithm 1 in the paper (UniInv)
        """
        sample, prompt, bs = self.align_bs(sample, prompt)
        model.set_prompt(prompt)
        timesteps, sigmas = model.get_schedule(num_steps=self.cfg.num_steps, height=height, width=width)
        
        # Inversion goes from t=0 (clean) to t=1 (noise)
        timesteps = torch.flip(timesteps, [0]) if isinstance(timesteps, torch.Tensor) else timesteps[::-1]
        
        latents = sample.clone()
        if return_trajectory:
            rets = [latents.float().clone().cpu()]
        v_pred_next = None 
        step_threshold = round(self.cfg.alpha * len(timesteps[:-1]))
        
        for i, (t_curr, t_prev) in tqdm(enumerate(zip(timesteps[:-1], timesteps[1:])), desc='Inverting with UniInv', disable=not verbose):
            if i >= step_threshold:
                continue
            # 1. First Euler Step
            if v_pred_next is None:
                if self.cfg.zero_init and i == 0:
                    v_pred = torch.zeros_like(latents)
                else:
                    v_pred = model.get_v_prediction(
                        latents,
                        t_curr,
                        guidance_scale = self.cfg.invert_guidance_scale,
                        cfg_strength = self.cfg.invert_cfg_strength,
                        height=height,
                        width=width
                    )
            else:
                v_pred = v_pred_next
            
            # Estimate next point (Euler)
            latents_next = latents + (t_prev - t_curr) * v_pred

            # 2. Correction Step (Second Order)
            v_pred_next = model.get_v_prediction(
                latents_next,
                t_prev,
                guidance_scale = self.cfg.invert_guidance_scale,
                cfg_strength = self.cfg.invert_cfg_strength,
                height=height,
                width=width
            )

            # Update with corrected velocity
            latents = latents + (t_prev - t_curr) * v_pred_next
            if return_trajectory:
                rets.append(latents.float().clone().cpu())

        if return_trajectory:
            return latents, rets
             
        return latents 
    
    def edit(
        self,
        inverted_latents: torch.Tensor,
        src_prompt: str | List[str],
        tgt_prompt: str | List[str],
        model: RFModel,
        height: int,
        width: int,
        verbose: bool = True
    ):
        """
        Algorithm 2 in the paper (Uni-Edit)
        """
        inverted_latents, tgt_prompt, bs = self.align_bs(inverted_latents, tgt_prompt)
        
        timesteps, sigmas = model.get_schedule(num_steps=self.cfg.num_steps, height=height, width=width)
        
        latents = inverted_latents.clone()
        
        # Calculate step threshold for skipping
        step_threshold = round(self.cfg.alpha * len(timesteps[:-1]))
        
        for i, (t_curr, t_prev) in tqdm(enumerate(zip(timesteps[:-1], timesteps[1:])), desc='Editing with UniEdit', disable=not verbose):
            
            # Skip early steps based on alpha
            if i < (len(timesteps[:-1]) - step_threshold):
                continue
            
            # 1. Get predictions for Target and Source
            
            # Target Prediction
            model.set_prompt(tgt_prompt)
            pred_trg = model.get_v_prediction(
                latents,
                t_curr,
                guidance_scale=self.cfg.edit_guidance_scale,
                cfg_strength=self.cfg.edit_cfg_strength,
                height=height,
                width=width
            )
            
            # Source Prediction
            model.set_prompt(src_prompt)
            pred_src = model.get_v_prediction(
                latents,
                t_curr,
                guidance_scale=self.cfg.edit_guidance_scale,
                cfg_strength=self.cfg.edit_cfg_strength,
                height=height,
                width=width
            )
            
            # 2. Calculate Mask (UniEdit Logic)
            cfg_component = pred_trg - pred_src
            
            if latents.ndim == 4: # (B, C, H, W)
                save_sub_map = cfg_component.abs().mean(dim=1, keepdim=True)
            else: # (B, N, C)
                save_sub_map = cfg_component.abs().mean(dim=-1, keepdim=True)
                
            # Normalize map to [0, 1]
            save_sub_map = (save_sub_map - save_sub_map.min()) / (save_sub_map.max() - save_sub_map.min() + 1e-8)
            
            # 3. Fuse Velocities
            fused_v = save_sub_map * pred_trg + (1 - save_sub_map) * pred_src
            
            # 4. Apply Omega Boost
            pred = fused_v + (save_sub_map + 1) * self.cfg.omega * cfg_component
            
            # 5. Update Latents (Euler)
            latents = latents + (t_prev - t_curr) * pred
            
        return latents

    @torch.no_grad()
    def __call__(
        self,
        src: torch.Tensor | List[torch.Tensor],
        model: RFModel,
        tgt_prompt: str | List[str],
        src_prompt: str | List[str] = "",
        **kwargs
    ):
        if isinstance(src, list):
            src = torch.stack(src, dim=0)
            
        height, width = src.shape[-2:]
        src = src.to(model.device)
        
        # Encode image to latents
        latents = model.encode(src.to(model.device))
        
        # 1. Inverting uses null text
        inv_prompt = "" 
        
        inverted_latents = self.invert(
            latents,
            inv_prompt,
            model,
            height,
            width
        )
        
        # 2. Edit
        edited_latents = self.edit(
            inverted_latents,
            src_prompt,
            tgt_prompt,
            model,
            height,
            width
        )
        
        return model.decode(edited_latents, height, width, output_type='pil')
