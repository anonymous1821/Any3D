"""
Implementation of FireFlow (https://github.com/HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing)
Please cite the original paper "FireFlow: Fast Inversion of Rectified Flow for Image Semantic Editing" (ICML 2025)
if you find this method helpful

In conclusion, FireFlow proposed:
An efficient way to approximate midpoint velocity in RF-Solver, with one NFE each step, to reduce the NFE of inversion. 
The Editing method exactly follows the RF-Solver setting 
"""
from .inversion import Inversion 
from dataclasses import dataclass 
from utils.registry import INVERSION_REGISTRY 
from utils.config import BaseConfig 
from model import RFModel, FLUXAttInjectModel
import torch 
from typing import * 
from tqdm import tqdm
from .RFSolver import RFSolver, RFSolverConfig

@dataclass 
class FireFlowConfig(RFSolverConfig):
    type: str = 'fireflow'

@INVERSION_REGISTRY.register('fireflow')
class FireFlow(RFSolver):
    Config = FireFlowConfig 

    def invert(
        self, 
        sample: torch.Tensor,
        model: RFModel, 
        prompt: str | List[str] = None,
        height: int = None, 
        width: int = None, 
        cond: Dict[str, torch.Tensor] = None,
        pos_cond: Dict[str, torch.Tensor] = None,
        verbose: bool = True,
        return_trajectory: bool = False,
        **kwargs
    ):
        device = model.device
        sample, prompt, bs = self.align_bs(sample, prompt)
        if cond is None and pos_cond is None:
            model.set_prompt(prompt)
        N = self.cfg.num_steps 
        timesteps, _ = model.get_schedule(num_steps=N, height=height, width=width)
        timesteps = torch.flip(timesteps, [0]) if isinstance(timesteps, torch.Tensor) else timesteps[::-1]
        latents = sample.clone()
        if return_trajectory:
            rets = [latents.float().clone().cpu()]
        next_v_pred = None
        for i, (t_curr, t_prev) in tqdm(enumerate(zip(timesteps[:-1], timesteps[1:])), desc='Inverting by FireFlow', disable=not verbose):
            if next_v_pred is None:
                v_pred = model.get_v_prediction(
                    latents=latents,
                    t=t_curr,
                    cfg_strength=self.cfg.invert_cfg_strength,
                    guidance_scale=self.cfg.invert_guidance_scale,
                    height=height,
                    width=width,
                    cond=cond,
                    pos_cond=pos_cond,
                    **kwargs
                )
            else:
                v_pred = next_v_pred
            
            mid_latents = latents + (t_prev - t_curr) / 2 * v_pred 
            mid_t = t_curr + (t_prev - t_curr) / 2
            v_pred_mid = model.get_v_prediction(
                latents=mid_latents,
                t=mid_t,
                cfg_strength=self.cfg.invert_cfg_strength,
                guidance_scale=self.cfg.invert_guidance_scale,
                height=height,
                width=width,
                cond=cond,
                pos_cond=pos_cond,
                **kwargs
            )

            next_v_pred = v_pred_mid

            latents = latents + (t_prev - t_curr) * v_pred_mid
            if return_trajectory:
                rets.append(latents.float().clone().cpu())
        
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
        model.set_prompt(prompt)
        N = self.cfg.num_steps 
        timesteps, _ = model.get_schedule(num_steps=N, height=height, width=width)
        
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
        
        next_v_pred = None

        for i, (t_curr, t_prev) in tqdm(enumerate(zip(timesteps[:-1], timesteps[1:])), desc='Inverting by FireFlow', disable=not verbose):
            # Determine injection requirements
            inject = inject_list[i]
            inject_next = inject_list[i+1] if i + 1 < len(inject_list) else False

            info['t'] = timesteps_list_inv[i+1]
            info['inverse'] = True
            info['second_order'] = False
            info['inject'] = inject
            
            if next_v_pred is None:
                v_pred, info = model.get_v_prediction(
                    latents=latents,
                    t=t_curr,
                    cfg_strength=self.cfg.invert_cfg_strength,
                    guidance_scale=self.cfg.invert_guidance_scale,
                    height=height,
                    width=width,
                    info=info
                )
            else:
                v_pred = next_v_pred
                # Copy attention features
                # We copy from the previous step's midpoint (second_order=True) 
                # to current step's start (second_order=False)
                prev_t_val = timesteps_list_inv[i]
                curr_t_val = timesteps_list_inv[i+1]
                
                prefix = str(prev_t_val) + '_True'
                new_prefix = str(curr_t_val) + '_False'
                
                for k in list(info['feature'].keys()):
                    if k.startswith(prefix):
                        suffix = k[len(prefix):]
                        new_key = new_prefix + suffix
                        info['feature'][new_key] = info['feature'][k]

            mid_latents = latents + (t_prev - t_curr) / 2 * v_pred 
            mid_t = t_curr + (t_prev - t_curr) / 2
            
            info['second_order'] = True
            # Force inject if next step needs it for reuse
            info['inject'] = inject or inject_next
            
            v_pred_mid, info = model.get_v_prediction(
                x_t=mid_latents,
                t=mid_t,
                cfg_strength=self.cfg.invert_cfg_strength,
                guidance_scale=self.cfg.invert_guidance_scale,
                height=height,
                width=width,
                info=info
            )
            
            next_v_pred = v_pred_mid
            latents = latents + (t_prev - t_curr) * v_pred_mid
            
            if return_trajectory:
                rets.append(latents.clone().cpu().numpy())
        
        if return_trajectory:
            return latents, info, rets
        
        return latents, info