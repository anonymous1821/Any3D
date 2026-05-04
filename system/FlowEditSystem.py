"""
Implementation of FlowEdit (https://github.com/fallenshock/FlowEdit)
Please cite the original paper "FlowEdit: Inversion-Free Text-Based Editing Using Pretrained Flow Models" (ICCV 2025)
if you find this method helpful

In conclusion, FlowEdit proposed:
1. Reinterpretation of Editing by Inversion as a direct path between the source and target distributions
2. A heuristic method mapping each mode in the source distribution to its nearest mode in the target
distribution, specifically tested by image editing. 
"""
from utils.registry import SYSTEM_REGISTRY
from utils.factory import instantiate_model 
import torch 
from utils.config import BaseConfig 
from dataclasses import dataclass 
from utils.common import seed_everything
from typing import * 
from utils.logger import setup_logging, get_logger
setup_logging()
log = get_logger(__name__)

@dataclass 
class FlowEditConfig(BaseConfig):
    Model: BaseConfig 
    num_steps: int = 28
    n_avg: int = 1 
    src_guidance_scale: float = 1.5 
    src_cfg_strength: float = 1.5 
    tar_guidance_scale: float = 5.5 
    tar_cfg_strength: float = 5.5 
    n_min: int = 0 
    n_max: int = 24
    type: str = 'flow-edit'
    seed: Optional[int] = None

@SYSTEM_REGISTRY.register('flow-edit')
class FlowEditSystem:
    Config = FlowEditConfig
    def __init__(self, cfg: FlowEditConfig, model=None):
        self.cfg = cfg 
        if self.cfg.seed is not None:
            seed_everything(self.cfg.seed)
        if model is not None:
            log.info(f'Initializing FLowEdit System with given model: {type(model)}')
        else:
            log.info(f"Initializing FLowEdit System with new model instantiated: {cfg.Model['type']}")
        self.model = model or instantiate_model(cfg.Model)
    
    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self

    @torch.no_grad()
    def __call__(
        self,
        source: torch.Tensor | List[torch.Tensor] | None = None,
        src_prompt: str | List[str] = None,
        tgt_prompt: str | List[str] = None,
        src_latents: torch.Tensor | List[torch.Tensor] = None,
        tgt_latents: torch.Tensor | List[torch.Tensor] = None,
        return_latent: bool = False,
        src_cond: dict = None, 
        src_pos_cond: dict = None,
        tgt_cond: dict = None, 
        tgt_pos_cond: dict = None,
        src_params: dict = None, 
        tgt_params: dict = None,
        **kwargs
    ):
        """
        Currently, for Trellis2, make sure latents is provided and return_latents is True
        """
        if src_latents is not None:
            src_latents = [src_latents]
            x_src = torch.stack(src_latents, dim=0).to(self.model.device, dtype = self.model.dtype)
        else:
            if not isinstance(source, list):
                source = [source]
            source = [s.to(self.model.device, self.model.dtype) for s in source]
            width, height = source[0].shape[-2:]
            kwargs.update({'width': width, 'height': height})
            x_src = self.model.encode(torch.stack(source, dim=0))
    
        if src_cond is None:
            src_cond, src_pos_cond = self.model.set_prompt(src_prompt)
        if tgt_cond is None:
            tgt_cond, tgt_pos_cond = self.model.set_prompt(tgt_prompt)

        timesteps, sigmas = self.model.get_schedule(num_steps=self.cfg.num_steps, **kwargs)
        zt_edit = x_src.clone()
        num_steps = self.cfg.num_steps
        for i, t in enumerate(timesteps[:-1]):
            if num_steps - i > self.cfg.n_max:
                continue 
            t_i = sigmas[i]    
            if i < len(timesteps):
                t_im1 = sigmas[i+1]
            else:
                t_im1 = t_i 

            if num_steps - i > self.cfg.n_min:
                
                v_delta_avg = torch.zeros_like(x_src)
                
                for k in range(self.cfg.n_avg):             
                    fwd_noise = torch.randn_like(x_src).to(self.model.device, self.model.dtype)
                    zt_src = x_src * (1.0 - float(t_i)) + fwd_noise * float(t_i)
                    zt_tar = zt_edit + zt_src - x_src

                    vt_src = self.model.get_v_prediction(
                        latents=zt_src, 
                        t=t,
                        cfg_strength=self.cfg.src_cfg_strength,
                        guidance_scale=self.cfg.src_guidance_scale,
                        cond=src_cond,
                        pos_cond=src_pos_cond,
                        **src_params if src_params is not None else {},
                        **kwargs
                    )

                    vt_tar = self.model.get_v_prediction(
                        latents=zt_tar, 
                        t=t,
                        cfg_strength=self.cfg.tar_cfg_strength,
                        guidance_scale=self.cfg.tar_guidance_scale,
                        cond=tgt_cond,
                        pos_cond=tgt_pos_cond,
                        **tgt_params if tgt_params is not None else {},
                        **kwargs
                    )

                    # Debug: print norms for sparse latents to inspect whether vt_tar and vt_src differ
                    try:
                        src_norm = vt_src.norm().item()
                        tar_norm = vt_tar.norm().item()
                        delta_norm = (vt_tar - vt_src).norm().item()
                    except Exception as e:
                        print('flowedit per-step debug failed:', e)

                    v_delta_avg += (vt_tar - vt_src) * (1.0 / self.cfg.n_avg)
                zt_edit = zt_edit.float()
                zt_edit = zt_edit + v_delta_avg * (t_im1 - t_i)
                try:
                    zt_edit = zt_edit.to(self.model.dtype)
                except:
                    pass
            
            else:
                if i == num_steps - self.cfg.n_min:
                    fwd_noise = torch.randn_like(x_src).to(self.model.device, self.model.dtype)
                    xt_src = x_src * (1.0 - float(t)) + fwd_noise * float(t)
                    xt_tar = zt_edit + xt_src - x_src

                vt_tar = self.model.get_v_prediction(
                    latents=xt_tar, 
                    t=t,
                    cfg_strength=self.cfg.tar_cfg_strength,
                    guidance_scale=self.cfg.tar_guidance_scale,
                    cond=tgt_cond,
                    pos_cond=tgt_pos_cond,
                    **tgt_params if tgt_params is not None else {},
                    **kwargs
                )

                xt_tar = xt_tar.float()
                xt_tar = xt_tar + vt_tar * (t_im1 - t_i)
                try:
                    xt_tar = xt_tar.to(self.model.dtype)
                except:
                    pass
        out = zt_edit if self.cfg.n_min == 0 else xt_tar
        if return_latent:
            return out
        return self.model.decode(out, output_type='pil', **kwargs)
