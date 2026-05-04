from utils.registry import SYSTEM_REGISTRY
from utils.factory import instantiate_model, instantiate_inversion 
import torch 
from utils.config import BaseConfig 
from utils.common import seed_everything
from dataclasses import dataclass 
from typing import * 
@dataclass 
class FlowEditInpaintConfig(BaseConfig):
    Model: BaseConfig 
    num_steps: int = 28
    n_avg: int = 1 
    src_guidance_scale: float = 1.5 
    src_cfg_strength: float = 1.5 
    tar_guidance_scale: float = 5.5 
    tar_cfg_strength: float = 5.5 
    n_min: int = 0 
    n_max: int = 24
    scale_by_inversion: bool = False 
    Inversion: BaseConfig = None
    type: str = 'flow-edit-inpaint'
    seed: Optional[int] = None

@SYSTEM_REGISTRY.register('flow-edit-inpaint')
class FlowEditInpaintSystem:
    Config = FlowEditInpaintConfig

    def __init__(self, cfg: FlowEditInpaintConfig, model=None, inversion=None):
        self.cfg = cfg 
        if self.cfg.seed is not None:
            seed_everything(self.cfg.seed)
        self.model = model or instantiate_model(cfg.Model)
        if cfg.scale_by_inversion:
            self.inversion = inversion or instantiate_inversion(cfg.Inversion)
        else:
            self.inversion = None 

    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self
    
    @torch.no_grad()
    def __call__(
        self,
        source: torch.Tensor | List[torch.Tensor] = None,
        mask: torch.Tensor | List[torch.Tensor] = None,
        latents_mask: torch.Tensor | List[torch.Tensor] = None,
        src_prompt: str | List[str] = None, 
        tgt_prompt: str | List[str] = None,
        src_latents: torch.Tensor | List[torch.Tensor] = None,
        tgt_latents: torch.Tensor | List[torch.Tensor] = None,
        src_cond: dict = None, 
        src_pos_cond: dict = None,
        tgt_cond: dict = None, 
        tgt_pos_cond: dict = None,
        src_params: dict = None, 
        tgt_params: dict = None,
        inpaint_image: torch.Tensor | List[torch.Tensor] = None,
        inpaint_latents: torch.Tensor | List[torch.Tensor] = None,
        return_latent: bool = False,
        **kwargs
    ):
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
    
        if tgt_latents is not None:
            tgt_latents = [tgt_latents]
            fwd_noise = torch.stack(tgt_latents, dim=0).to(self.model.device, dtype = self.model.dtype)
        else:
            fwd_noise = torch.randn_like(x_src)
        if src_cond is None:
            src_cond, src_pos_cond = self.model.set_prompt(src_prompt)
        if tgt_cond is None:
            tgt_cond, tgt_pos_cond = self.model.set_prompt(tgt_prompt)

        if inpaint_latents is None:
            inpaint_image = source if not inpaint_image else inpaint_image     
            inpaint_latents = self.model.encode(inpaint_image)
        if latents_mask is None:
            mask = [mask] if isinstance(mask, torch.Tensor) else mask
            mask = self.model.get_latents_mask(mask, batch_size=len(source), **kwargs)
        else:
            mask = latents_mask
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
                    try:
                        fwd_noise = fwd_noise.to(self.model.device, self.model.dtype)
                    except:
                        fwd_noise = fwd_noise.to(self.model.device)
                    zt_src = (1-float(t_i))*x_src + float(t_i)*fwd_noise
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

                    v_delta_avg += (vt_tar - vt_src) * (1.0 / self.cfg.n_avg)

                zt_edit = zt_edit.float()
                zt_edit = zt_edit + v_delta_avg * (t_im1 - t_i) 
                try:
                    zt_edit = zt_edit.to(self.model.dtype)
                except:
                    pass
                zt_edit = zt_edit * mask + inpaint_latents * (1 - mask)
            
            else:
                if i == num_steps - self.cfg.n_min:
                    fwd_noise = torch.randn_like(x_src)
                    xt_src = (1-float(t))*x_src + float(t)*fwd_noise
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
                xt_tar = xt_tar * mask + inpaint_latents * (1 - mask)
        out = zt_edit if self.cfg.n_min == 0 else xt_tar
        if return_latent: 
            return out 
        return self.model.decode(out, output_type='pil', **kwargs)
