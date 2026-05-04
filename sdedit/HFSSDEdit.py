"""
Implementation of HSF-SDEdit (https://github.com/ryunuri/Elevate3D)
Please cite the original paper "Elevating 3D Models: High-Quality Texture and Geometry Refinement from a Low-Quality Model" (SIGGRAPH 2025)
if you find this algorithm helpful 

In conclusion, HSF-SDEdit proposed:
Employ a frequency bandit to select the low frequency texture to edit and keep the high frequency area, 
to trade off between quality and fidelity.
Note that the paper additionally proposed a pipeline for refining generated 3D model using the 
HSF-SDEdit. 

Note that since HSF-SDEdit require converting image to frequency domain to compute 
the mask, this is not directly applicable to 3D domain. 
"""
from model import RFModel
from typing import List 
from .SDEdit import SDEdit, SDEditConfig
from dataclasses import dataclass
from utils.registry import SDEDIT_REGISTRY
from torch.nn import functional as F
import torch 
from utils.fft import (
    extract_low_frequency_and_mask,
    extract_high_frequency
)
@dataclass
class HFSSDEditConfig(SDEditConfig):
    low_freq_ratio: float = 0.0625
    replace_steps: int = 20

@SDEDIT_REGISTRY.register('hfs-sdedit')
class HFSSDEdit(SDEdit):
    Config = HFSSDEditConfig

    def __init__(self, cfg: HFSSDEditConfig):
        super().__init__(cfg)
    
    def sample_step(
        self,
        model: RFModel , 
        latents:torch.Tensor, 
        image_latents: torch.Tensor, 
        noise: torch.Tensor,
        t_prev: torch.Tensor, 
        t_curr: torch.Tensor,
        height: int,
        width: int, 
        latent_height: int,
        latent_width: int,
        num_channels_latents: int,
        mix: bool = True,
        **kwargs
    ):
        bs = latents.shape[0]
        v_pred = model.get_v_prediction(
            latents = latents,
            t = t_curr,
            guidance_scale = self.cfg.guidance_scale, 
            cfg_strength = self.cfg.cfg_strength,
            height = height,
            width = width,
            **kwargs
        )
        latents = latents + (t_curr - t_prev) * v_pred
        
        if mix:
            #TODO:
            # Some Model like QwenImage use a vae trained jointly on image&video, resulting in extra time dimension
            # Record the exact latents shape for different model integration.
            latents_shape = latents.shape
            # original image latents at current noise level 
            noised_image_latents = t_prev * noise + (1 - t_prev) * image_latents
            noised_image_latents = model._unpack_latents(latents=noised_image_latents, num_channels_latents=model.num_channels_latents, height=height, width=width).reshape(bs, num_channels_latents, latent_height, latent_width)
            
            # Extract high frequencies from noised image
            im_high_latents = extract_high_frequency(noised_image_latents.float(), low_freq_ratio=self.cfg.low_freq_ratio).to(latents.dtype)

            latents = model._unpack_latents(latents = latents, num_channels_latents = model.num_channels_latents, height = height, width = width).reshape(bs, num_channels_latents, latent_height, latent_width)
            
            low_latents, _ = extract_low_frequency_and_mask(latents.float(), low_freq_ratio=self.cfg.low_freq_ratio)
            
            # Swap high frequencies
            latents = low_latents.to(latents.dtype) + im_high_latents
            # latents = latents.reshape(latents_shape)
            latents = model._pack_latents(latents = latents, num_channels_latents = model.num_channels_latents, height = height, width = width).to(model.dtype)

        return latents 

    def sample(
        self,
        model: RFModel,
        perturbed_latents: torch.Tensor,
        image_latents: torch.Tensor,
        prompt: str | List[str], 
        height: int, 
        width: int, 
        negative_prompt: str | List[str] = "",
        timesteps: List[torch.Tensor] = None,
        mask: torch.Tensor | List[torch.Tensor]= None, # 1 for editied region, 0 for static region
        inpaint_latents: torch.Tensor | List[torch.Tensor] = None, 
        **kwargs
    ):
        model.set_prompt(prompt, negative_prompt)
        latents = perturbed_latents.clone()
        timesteps = model.get_schedule(num_steps=self.cfg.num_steps, height=height, width=width)[0] if timesteps is None else timesteps
        start = int((1 - self.cfg.strength) * len(timesteps))

        latent_height = 2 * (int(height) // (model.vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (model.vae_scale_factor * 2))
        bs = latents.shape[0]

        noise = torch.randn_like(image_latents)

        # Prepare inpainting noise
        inpaint_noise = None
        if mask is not None:
            mask = [mask] if not isinstance(mask, list) else mask
            mask = [m.to(model.device, model.dtype) for m in mask]
            mask = model.get_latents_mask(mask, bs, height, width)
            if isinstance(inpaint_latents, torch.Tensor):
                inpaint_noise = torch.randn_like(inpaint_latents)

        for i, (t_prev, t_curr) in enumerate(zip(timesteps[start:-1], timesteps[start+1:])):
            # Inpainting 
            # If a single tensor is provided, we scale it to the same noise level by simply adding noise
            if mask is not None and isinstance(inpaint_latents, torch.Tensor):
                noisy_inpaint_latents = t_curr * inpaint_noise + (1 - t_curr) * inpaint_latents
                latents = latents * mask + noisy_inpaint_latents * (1 - mask)

            # If a list of latents is provided, they already correspond to different noise levels (possibly from inversion)
            if mask is not None and isinstance(inpaint_latents, list):
                latents = latents * mask + inpaint_latents[i].to(model.device, model.dtype) * (1 - mask)
            
            latents = self.sample_step(
                model=model,
                latents=latents, 
                image_latents=image_latents, 
                noise=noise,
                t_prev=t_prev, 
                t_curr=t_curr,
                height=height,
                width=width,
                latent_height=latent_height,
                latent_width=latent_width,
                num_channels_latents=model.num_channels_latents,
                mix=(i + 1) < self.cfg.replace_steps,
                **kwargs
            )

        return model.decode(latents, height, width, output_type='pil')

    def __call__(
        self,
        src: torch.Tensor | List[torch.Tensor], 
        model: RFModel, 
        prompt: str | List[str],
        negative_prompt: str | List[str] = "",
        mask: torch.Tensor | List[torch.Tensor]= None, # 1 for editied region, 0 for static region
        inpaint_latents: torch.Tensor | List[torch.Tensor] = None, 
        **kwargs
    ):
        height, width = src.shape[-2:] if not isinstance(src, list) else src[0].shape[-2:]
        src = model.encode(src)
        timesteps, _ = model.get_schedule(num_steps=self.cfg.num_steps, height=height, width=width)
        perturbed_latents = self.perturb(src, timesteps)
        return self.sample(
            model = model,
            perturbed_latents = perturbed_latents,
            image_latents = src,
            prompt = prompt,
            height = height,
            width = width,
            negative_prompt = negative_prompt,
            timesteps = timesteps,
            mask = mask,
            inpaint_latents = inpaint_latents,
            **kwargs
        )