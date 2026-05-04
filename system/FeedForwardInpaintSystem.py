"""
Incorporate inpainting mechanism for feed forward system. Inpainting requires
converting latents into corresponding noise level at each step. Naively this can be
done by adding noise, but also supported by inversion trajectory.
"""
from utils.registry import SYSTEM_REGISTRY 
from utils.factory import (
    instantiate_model,
    instantiate_inversion
)
import torch 
from utils.config import BaseConfig 
from dataclasses import dataclass 
from utils.common import seed_everything
from typing import * 
from model import RFModel 

@dataclass 
class FeedForwardInpaintConfig(BaseConfig):
    Model: BaseConfig 
    num_steps: int = 50
    cfg_strength: float = 4.0 
    guidance_scale: float = 4.0
    scale_by_inversion: bool = False 
    Inversion: BaseConfig = None 
    type: str = 'feed-forward-inpaint'
    seed: Optional[int] = None

@SYSTEM_REGISTRY.register('feed-forward-inpaint')
class FeedForwardInpaintSystem:
    Config = FeedForwardInpaintConfig

    def __init__(self, cfg: FeedForwardInpaintConfig, model: RFModel=None):
        self.cfg = cfg 
        if self.cfg.seed is not None:
            seed_everything(self.cfg.seed)
        self.model = model or instantiate_model(cfg.Model)
        if cfg.scale_by_inversion:
            self.inversion = instantiate_inversion(cfg.Inversion)
        else:
            self.inversion = None 
    
    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self
    
    def set_prompt(
        self, 
        prompt: str | List[str],
        negative_prompt: str | List[str] = "",
        image: torch.Tensor | List[torch.Tensor] = None,
        **kwargs
    ):
        return self.model.set_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image
        )
    
    @torch.no_grad()
    def __call__(
        self, 
        source: torch.Tensor | List[torch.Tensor],
        mask: torch.Tensor | List[torch.Tensor], 
        prompt: str | List[str] = None,
        inversion_prompt: str | List[str] = " ",
        negative_prompt: str | List[str] = "",
        inpaint_image: torch.Tensor | List[torch.Tensor] = None,
        inpaint_latents: torch.Tensor | List[torch.Tensor] = None,
        height: int = None, 
        width: int = None,
        cond: Dict = None, 
        pos_cond: Dict = None,
        verbose: bool = True,
        **kwargs
    ):
        assert (cond is not None) or (prompt is not None), "Either cond or prompt must be provided"
        if not isinstance(source, list):
            source = [source]
        input_height, input_width = source[0].shape[-2:]
        height = height or input_height 
        width = width or input_width

        if inpaint_image is not None and not isinstance(inpaint_image, list):
            inpaint_image = [inpaint_image]

        if inpaint_latents is None:
            inpaint_image = source if inpaint_image is None else inpaint_image     
            inpaint_latents = self.model.encode(inpaint_image)
        
        #TODO: Refine the snippet 
        if self.inversion is not None:
            # TODO: Support QwenImageEdit 
            # dummy_image = [torch.zeros_like(inpaint_image[0])] #NOTE: the image prompt
            # extra_kwargs = {}
            # condition_images, vae_images, vae_image_sizes = self.model.preprocess_image(dummy_image)
            # invert_cond, invert_pos_cond = self.model.set_prompt(
            #     prompt=prompt, 
            #     negative_prompt=negative_prompt,
            #     image=condition_images
            # )
            # image_latents = self.model.prepare_image_latents(vae_images, height, width, generator=None)
            # extra_kwargs['image_latents'] = image_latents
            # extra_kwargs['vae_image_sizes'] = vae_image_sizes

            # _, inpaint_latents = self.inversion.invert(
            #     sample=inpaint_latents,
            #     prompt=inversion_prompt,
            #     model=self.model, 
            #     height=height,
            #     width=width,
            #     cond=invert_cond,
            #     pos_cond=invert_pos_cond,
            #     return_trajectory=True,
            #     verbose=verbose,
            #     **extra_kwargs
            # )
            _, inpaint_latents = self.inversion.invert(
                sample=inpaint_latents,
                prompt=inversion_prompt, #NOTE: Use what prompt for inversion? 
                model=self.model, 
                height=height,
                width=width,
                return_trajectory=True,
                verbose=verbose,
            )
            inpaint_latents = inpaint_latents[::-1]
        
        return self.model.sample(
            prompt=prompt, 
            image=source,
            negative_prompt=negative_prompt,
            num_steps=self.cfg.num_steps,
            cfg_strength=self.cfg.cfg_strength,
            guidance_scale=self.cfg.guidance_scale,
            mask=mask,
            inpaint_latents=inpaint_latents,
            height=height,
            width=width,
            inverted_latents=inpaint_latents,
            image_latents=source,
            cond=cond,
            pos_cond=pos_cond,
            verbose=verbose
        )