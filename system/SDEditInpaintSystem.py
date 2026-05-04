from utils.registry import SYSTEM_REGISTRY
from utils.factory import (
    instantiate_model,
    instantiate_sdedit,
    instantiate_inversion
)
from typing import List, Optional
import torch
from utils.config import BaseConfig
from utils.common import seed_everything
from dataclasses import dataclass
@dataclass
class SDEditConfig(BaseConfig):
    Model: BaseConfig
    SDEdit: BaseConfig
    scale_by_inversion: bool = False 
    Inversion: BaseConfig = None 
    type: str = 'sde-edit'
    seed: Optional[int] = None

@SYSTEM_REGISTRY.register('sde-edit-inpaint')
class SDEditInpaintSystem:
    Config = SDEditConfig

    def __init__(self, cfg: SDEditConfig, model=None, sdedit=None, inversion=None):
        if cfg.seed is not None:
            seed_everything(cfg.seed)
        self.model = model or instantiate_model(cfg.Model)
        self.sdedit = sdedit or instantiate_sdedit(cfg.SDEdit)
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
        source: torch.Tensor | List[torch.Tensor],
        mask: torch.Tensor | List[torch.Tensor],
        prompt: str | List[str],
        negative_prompt: str | List[str] = "",
        inpaint_image: torch.Tensor | List[torch.Tensor] = None,
        inpaint_latents: torch.Tensor | List[torch.Tensor] = None,
        **kwargs
    ):
        if not isinstance(source, list):
            source = [source]
        height, width = source[0].shape[-2:]
        source = [s.to(self.model.device, self.model.dtype) for s in source]
        if inpaint_image is not None and not isinstance(inpaint_image, list):
            inpaint_image = inpaint_image.to(self.model.device, self.model.dtype)
            inpaint_image = [inpaint_image]

        if inpaint_latents is None:
            inpaint_image = source if inpaint_image is None else inpaint_image     
            inpaint_latents = self.model.encode(inpaint_image)
        
        if self.inversion is not None:
            _, inpaint_latents = self.inversion.invert(
                sample=inpaint_latents,
                prompt="", #NOTE: Use what prompt for inversion? 
                model=self.model, 
                height=height,
                width=width,
                return_trajectory=True
            )
            inpaint_latents = inpaint_latents[::-1]

        return self.sdedit(
            source,
            self.model,
            prompt,
            negative_prompt,
            mask = mask,
            inpaint_latents = inpaint_latents,
            **kwargs
        )