"""
Implementation of RF-Solver (https://github.com/wangjiangshan0725/RF-Solver-Edit)
Please cite the original paper "Taming Rectified Flow for Inversion and Editing" (ICML 2025)
if you find this method helpful

In conclusion, RF-Solver proposed
1. An inversion trajectory approximated by second order Taylor Expansion, solving the forward ODE more accurately
2. An editing method that inject the information of original image by injecting the attention value during inversion to attention layer during editing.
"""
from .inversion import Inversion
from dataclasses import dataclass 
from utils.registry import INVERSION_REGISTRY
from utils.config import BaseConfig
from utils.common import seed_everything
from model import RFModel, FLUXAttInjectModel
import torch 
from typing import * 
from tqdm import tqdm 

@dataclass
class RFSolverConfig(BaseConfig):
    num_inversion_steps: int = 10
    invert_cfg_strength: float = 1.0
    invert_guidance_scale: float = 1.0
    edit_cfg_strength: float = 1.0
    edit_guidance_scale: float = 5.0
    num_steps: int = 50
    inject_step: int = 10
    type: str = 'rf-solver'

@INVERSION_REGISTRY.register('rf-solver')
class RFSolver(Inversion):
    Config = RFSolverConfig

    def __init__(self, cfg: RFSolverConfig):
        self.cfg = cfg

    def invert(
        self,
        sample: torch.Tensor,
        prompt: str | List[str],
        model: RFModel,
        height: int,
        width: int,
        cond: Dict[str, torch.Tensor] = None,
        pos_cond: Dict[str, torch.Tensor] = None,
        verbose: bool = True,
        return_trajectory: bool = False,
        **kwargs
    ):
        device = model.device
        sample, prompt, bs = self.align_bs(sample, prompt)
        model.set_prompt(prompt)
        # We must use the same number of steps as editing to ensure the timesteps match
        N = self.cfg.num_steps 
        timesteps, _ = model.get_schedule(num_steps=N, height=height, width=width)
        timesteps = torch.flip(timesteps, [0])
        latents = sample.clone()
        if return_trajectory:
            rets = [latents.float().clone().cpu()]
        for i, (t_curr, t_prev) in tqdm(enumerate(zip(timesteps[:-1], timesteps[1:])), desc='Inverting by RF-Solver', disable=not verbose):
            v_pred = model.get_v_prediction(
                x_t=latents,
                t=t_curr,
                cfg_strength=self.cfg.invert_cfg_strength,
                guidance_scale=self.cfg.invert_guidance_scale,
                height=height,
                width=width,
                cond=cond,
                pos_cond=pos_cond,
                **kwargs
            )
            
            mid_latents = latents + (t_prev - t_curr) / 2 * v_pred 
            mid_t = t_curr + (t_prev - t_curr) / 2
            v_pred_mid = model.get_v_prediction(
                x_t=mid_latents,
                t=mid_t,
                cfg_strength=self.cfg.invert_cfg_strength,
                guidance_scale=self.cfg.invert_guidance_scale,
                height=height,
                width=width,
                cond=cond,
                pos_cond=pos_cond,
                **kwargs
            )

            first_order = (v_pred_mid - v_pred) / ((t_prev - t_curr) / 2)
            latents = latents + (t_prev - t_curr) * v_pred + 0.5 * (t_prev - t_curr) ** 2 * first_order
            if return_trajectory:
                rets.append(latents.clone().cpu().numpy())
        
        if return_trajectory:
            return latents, rets
        
        return latents

    def invert_attn_record(
        self,
        sample: torch.Tensor,
        prompt: str | List[str],
        model: FLUXAttInjectModel,
        height: int,
        width: int,
        verbose: bool = True,
        return_trajectory: bool = False
    ):
        """
        Inverting while recording the attention values, which is further used
        in proposed editing.
        """
        device = model.device
        sample, prompt, bs = self.align_bs(sample, prompt)
        if cond is None and pos_cond is None:
            model.set_prompt(prompt)
        # We must use the same number of steps as editing to ensure the timesteps match
        N = self.cfg.num_steps 
        timesteps, _ = model.get_schedule(num_steps=N, height=height, width=width)
        
        # Save timesteps as list for consistent float representation in info['t']
        timesteps_list = timesteps.tolist()
        info = {}
        info['feature'] = {}
        info['timesteps_list'] = timesteps_list

        timesteps = torch.flip(timesteps, [0])
        timesteps_list_inv = timesteps_list[::-1]
        
        latents = sample.clone()
        
        inject_step = min(self.cfg.inject_step, N)
        inject_list = [True] * inject_step + [False] * (N - inject_step)
        inject_list = inject_list[::-1]

        if return_trajectory:
            rets = []
        for i, (t_curr, t_prev) in tqdm(enumerate(zip(timesteps[:-1], timesteps[1:])), desc='Inverting by RF-Solver', disable=not verbose):
            info['t'] = timesteps_list_inv[i+1]
            info['inverse'] = True
            info['second_order'] = False
            info['inject'] = inject_list[i]
            
            v_pred, info = model.get_v_prediction(
                x_t=latents,
                t=t_curr,
                cfg_strength=self.cfg.invert_cfg_strength,
                guidance_scale=self.cfg.invert_guidance_scale,
                height=height,
                width=width,
                info=info
            )
            
            mid_latents = latents + (t_prev - t_curr) / 2 * v_pred 
            mid_t = t_curr + (t_prev - t_curr) / 2
            
            info['second_order'] = True
            
            v_pred_mid, info = model.get_v_prediction(
                x_t=mid_latents,
                t=mid_t,
                cfg_strength=self.cfg.invert_cfg_strength,
                guidance_scale=self.cfg.invert_guidance_scale,
                height=height,
                width=width,
                info=info
            )

            first_order = (v_pred_mid - v_pred) / ((t_prev - t_curr) / 2)
            latents = latents + (t_prev - t_curr) * v_pred + 0.5 * (t_prev - t_curr) ** 2 * first_order
            if return_trajectory:
                rets.append(latents.float().clone().cpu())
        
        if return_trajectory:
            return latents, info, rets
        
        return latents, info

    def edit(
        self,
        sample: torch.Tensor,
        prompt: str | List[str],
        model: FLUXAttInjectModel,
        info: Dict,
        height: int,
        width: int,
        verbose: bool = True,
    ):
        """
        Editing with the recorded attention values.
        """
        device = model.device
        sample, prompt, bs = self.align_bs(sample, prompt)
        model.set_prompt(prompt)
        N = self.cfg.num_steps
        timesteps, _ = model.get_schedule(num_steps=N, height=height, width=width)
        
        if 'timesteps_list' in info:
             timesteps_list = info['timesteps_list']
        else:
             timesteps_list = timesteps.tolist()

        latents = sample.clone()
        
        if 'inject_step' not in info:
            info['inject_step'] = self.cfg.inject_step
            
        inject_step = min(info['inject_step'], N)
        inject_list = [True] * inject_step + [False] * (N - inject_step)

        for i, (t_curr, t_prev) in tqdm(enumerate(zip(timesteps[:-1], timesteps[1:])), desc='Editing by RF-Solver', disable=not verbose):
            info['t'] = timesteps_list[i]
            info['inverse'] = False
            info['second_order'] = False
            info['inject'] = inject_list[i]
            
            v_pred, info = model.get_v_prediction(
                x_t=latents,
                t=t_curr,
                cfg_strength=self.cfg.edit_cfg_strength,
                guidance_scale=self.cfg.edit_guidance_scale,
                height=height,
                width=width,
                info=info
            )
            
            mid_latents = latents + (t_prev - t_curr) / 2 * v_pred 
            mid_t = t_curr + (t_prev - t_curr) / 2
            
            info['second_order'] = True
            
            v_pred_mid, info = model.get_v_prediction(
                x_t=mid_latents,
                t=mid_t,
                cfg_strength=self.cfg.edit_cfg_strength,
                guidance_scale=self.cfg.edit_guidance_scale,
                height=height,
                width=width,
                info=info
            )

            first_order = (v_pred_mid - v_pred) / ((t_prev - t_curr) / 2)
            latents = latents + (t_prev - t_curr) * v_pred + 0.5 * (t_prev - t_curr) ** 2 * first_order
        
        return latents

    @torch.no_grad()
    def __call__(
        self,
        src: torch.Tensor | List[torch.Tensor],
        model: FLUXAttInjectModel,
        src_prompt: str | List[str],
        tgt_prompt: str | List[str],
        **kwargs
    ):
        if isinstance(src, list):
            src = torch.stack(src, dim=0)
        height, width = src.shape[-2:]
        src = src.to(model.device)
        latents = model.encode(src.to(model.device))
        inverted_latents, info = self.invert_attn_record(
            sample=latents,
            prompt=src_prompt,
            model=model,
            height=height,
            width=width
        )
        edited_latents = self.edit(
            sample=inverted_latents,
            prompt=tgt_prompt,
            model=model,
            info=info,
            height=height,
            width=width
        )
        return model.decode(edited_latents, height, width, output_type='pil')
