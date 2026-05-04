from utils.registry import SYSTEM_REGISTRY
from utils.factory import (
    instantiate_model,
    instantiate_sdedit
)
from typing import List, Optional
import torch
from utils.config import BaseConfig
from utils.common import seed_everything
from dataclasses import dataclass
from utils.logger import setup_logging, get_logger
setup_logging()
log = get_logger(__name__)

@dataclass
class SDEditConfig(BaseConfig):
    Model: BaseConfig
    SDEdit: BaseConfig
    type: str = 'sde-edit'
    seed: Optional[int] = None

@SYSTEM_REGISTRY.register('sde-edit')
class SDEditSystem:
    Config = SDEditConfig

    def __init__(self, cfg: SDEditConfig, model=None, sdedit=None):
        if cfg.seed is not None:
            seed_everything(cfg.seed)
        if model is not None:
            log.info(f'Initializing FLowEdit System with given model: {type(model)}')
        else:
            log.info(f"Initializing FLowEdit System with new model instantiated: {cfg.Model['type']}")
        self.model = model or instantiate_model(cfg.Model)
        self.sdedit = sdedit or instantiate_sdedit(cfg.SDEdit)
        
    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self
        
    @torch.no_grad()
    def __call__(
        self,
        src: torch.Tensor | List[torch.Tensor] = None,
        src_latents: torch.Tensor | List[torch.Tensor] = None,
        prompt: str | List[str] = None,
        negative_prompt: str | List[str] = "",
        **kwargs
    ):
        
        
        if isinstance(src, torch.Tensor):
            src = [src]
        if isinstance(src, list):
            src = [s.to(self.model.device, self.model.dtype) for s in src]
        if isinstance(src_latents, torch.Tensor):
            src_latents = [src_latents]
        if isinstance(src_latents, list):
            src_latents = [s.to(self.model.device, self.model.dtype) for s in src_latents]
        return self.sdedit(
            src=src,
            src_latents=src_latents,
            model=self.model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            **kwargs
        )
