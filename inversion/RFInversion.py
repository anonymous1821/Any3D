"""
Implementation of RF-Inversion (https://rf-inversion.github.io/)
Please cite the original paper "Semantic Image Inversion and Editing using Stochastic Rectified Differential Equations" (ICLR 2025)
if you find this method helpful.

In conclusion, RF-Inversion proposed:
* A forward ODE to invert the given image
* A reverse ODE guided by the solution of a Linear Quadratic Regulator (LQR) problem, to edit the inverted image
* An equivalent reverse SDE (whose marginal distribution is the reverse ODE) which is more robust to initial image
"""
from .inversion import Inversion
from dataclasses import dataclass 
from utils.registry import INVERSION_REGISTRY
from utils.config import BaseConfig
from model import RFModel 
from tqdm import tqdm 
import torch 
from typing import * 
@dataclass
class RFInversionConfig(BaseConfig):
    num_inversion_steps: int = 28
    invert_guidance_scale: float = 0.
    edit_guidance_scale: float = 3.5 # Guidance distilled model (FLUX.1 Dev) use guidance_scale
    invert_cfg_strength: float = 1.0 
    edit_cfg_strength: float = 3.5 # Not guidance distilled model (QwenImage) use cfg_strength for true cfg.
    gamma: float = 0.5
    num_steps: int = 28
    start_timestep: float = 0.0
    stop_timestep: float = 7/28
    eta: float = 0.9
    decay_eta: bool = False
    eta_decay_power: float = 1.0
    enable_sde: bool = True

@INVERSION_REGISTRY.register('rf-inversion')
class RFInversion(Inversion):
    Config = RFInversionConfig

    def __init__(self, cfg: RFInversionConfig):
        self.cfg = cfg

    
    def invert(
        self,
        sample,
        prompt,
        model,
        height, 
        width,
        cond = None,
        pos_cond = None,
        verbose = True,
        return_trajectory: bool = False,
        **kwargs
        ):
        """
        Algorithm 1 in the paper 
        """
        device = model.device
        sample, prompt, bs = self.align_bs(sample, prompt)
        if cond is None and pos_cond is None:
            model.set_prompt(prompt)
        timesteps, sigmas = model.get_schedule(num_steps=self.cfg.num_inversion_steps, height=height, width=width)
        # Exclude the last sigma (0)
        N = len(sigmas) - 1
        Y_t = sample 
        y_1 = torch.randn_like(Y_t)
        if return_trajectory:
            rets = []
        for i in tqdm(range(N), desc='Inverting with RF-Inversion', disable=not verbose):
            t_i = torch.tensor(i / N, dtype=Y_t.dtype, device=device)

            v_uncond = model.get_v_prediction(
                latents=Y_t, 
                t=t_i, 
                cfg_strength = self.cfg.invert_cfg_strength,
                guidance_scale = self.cfg.invert_guidance_scale,
                height = height,
                width = width,
                cond = cond,
                pos_cond = pos_cond,
                **kwargs
            )
            v_cond = (y_1 - Y_t) / (1 - t_i)

            # Eq8 in original paper 
            v_hat = v_uncond + self.cfg.gamma * (v_cond - v_uncond)
            Y_t = Y_t + v_hat * (sigmas[i] - sigmas[i + 1])
            if return_trajectory:
                rets.append(Y_t.float().clone().cpu())
        
        if return_trajectory:
            return Y_t, rets
        return Y_t

    def edit_step(
        self,
        model: RFModel, 
        latents: torch.Tensor,
        y0: torch.Tensor,
        i: int,
        t: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        height: int,
        width: int,
        controlled_ode = True,
        **kwargs
    ):
        v_pred = model.get_v_prediction(
            latents, 
            t, 
            cfg_strength = self.cfg.edit_cfg_strength,
            guidance_scale = self.cfg.edit_guidance_scale,
            height=height,
            width=width,
            **kwargs
        )

        v_t = -v_pred 
        v_t_cond = (y0 - latents) / t
        eta_t = self.cfg.eta * controlled_ode
        if self.cfg.decay_eta:
            eta_t = eta_t * (1 - i / self.cfg.num_steps) ** self.cfg.eta_decay_power
        v_hat_t = v_t + eta_t * (v_t_cond - v_t)
        if not self.cfg.enable_sde: 
            latents = latents + v_hat_t * (sigma- sigma_next)
        else:
            if i == 0:
                drift = v_hat_t
                diffusion_coeff = 0
            else:
                drift = 2 * v_hat_t - latents / (1 - sigma)
                diffusion_coeff = (2 * sigma / (1 - sigma) * (sigma - sigma_next)).sqrt()
            latents = latents + (sigma - sigma_next) * drift + diffusion_coeff * torch.randn_like(latents)
        
        return latents
    
    def edit(
        self, 
        prompt: str | List[str],
        inverted_latents: torch.Tensor, # y1
        image_latents: torch.Tensor,  # y0
        model: RFModel,
        height: int, 
        width: int
    ):
        """
        Algorithm 2 in the paper 
        """
        inverted_latents, prompt, bs = self.align_bs(inverted_latents, prompt)
        model.set_prompt(prompt)
        timesteps, sigmas = model.get_schedule(num_steps=self.cfg.num_steps, height=height, width=width)
        start_timestep = int(self.cfg.start_timestep * self.cfg.num_steps)
        stop_timestep = min(int(self.cfg.stop_timestep * self.cfg.num_steps), self.cfg.num_steps)
        
        y0 = image_latents.clone()
        latents = inverted_latents.clone()
        N = len(sigmas)

        # Exclude t = 0
        for i, t in enumerate(timesteps[:-1]): 
            latents = self.edit_step(
                model=model, 
                latents=latents,
                y0=y0,
                i=i,
                t=t,
                sigma=sigmas[i],
                sigma_next=sigmas[i + 1],
                height=height,
                width=width,
                controlled_ode = (start_timestep <= i < stop_timestep)
            )
        return latents 

    @torch.no_grad()
    def __call__(
        self,
        src: torch.Tensor | List[torch.Tensor],
        model: RFModel,
        tgt_prompt: str | List[str], 
        **kwargs
    ):
        # RFInversion use null text for inversion
        src_prompt = ""
        if isinstance(src, list):
            src = torch.stack(src, dim=0)
        height, width = src.shape[-2:]
        src = src.to(model.device)
        latents = model.encode(src.to(model.device))
        inverted_latents = self.invert(
            latents, 
            src_prompt,
            model,
            height, 
            width
        )
        edited_latents = self.edit(
            tgt_prompt,
            inverted_latents,
            latents,
            model,
            height, 
            width
        )
        return model.decode(edited_latents, height, width, output_type='pil')
