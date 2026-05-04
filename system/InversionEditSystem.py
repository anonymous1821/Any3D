"""
This ImageBaseSystem is genearlly for image editing directly use the editing method proposed in 
the original inversion paper. While these papers mainly aimed at proposing an efficient
inversion technique, they also provide a method for editing coupled with the proposed inversion.

Done by simply calling inversion.
"""
from utils.registry import SYSTEM_REGISTRY
from utils.factory import (
    instantiate_model,
    instantiate_inversion
)
from typing import List, Optional
import torch
from utils.config import BaseConfig
from dataclasses import dataclass
from utils.common import seed_everything
@dataclass
class InversionEditConfig(BaseConfig):
    Model: BaseConfig
    Inversion: BaseConfig
    type: str = 'inversion-edit'
    seed: Optional[int] = None

@SYSTEM_REGISTRY.register('inversion-edit')
class InversionEditSystem:
    Config = InversionEditConfig

    def __init__(self, cfg: InversionEditConfig, model=None, inversion=None):
        if cfg.seed is not None:
            seed_everything(cfg.seed)
        self.model = model or instantiate_model(cfg.Model)
        self.inversion = inversion or instantiate_inversion(cfg.Inversion)
        
    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self
        
    def __call__(
        self,
        source: torch.Tensor | List[torch.Tensor],
        src_prompt: str | List[str], 
        tgt_prompt: str | List[str],
        **kwargs
    ):
        if not isinstance(source, list):
            source = [source]
        source = [s.to(self.model.device, self.model.dtype) for s in source]
        return self.inversion(
            src=source,
            model=self.model,
            src_prompt=src_prompt,
            tgt_prompt=tgt_prompt
        )