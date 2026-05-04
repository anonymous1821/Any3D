"""
This is a high level system that call the trained end-to-end editing models 
in feed forward manner 
"""
from utils.registry import SYSTEM_REGISTRY 
from utils.factory import instantiate_model 
import torch 
from utils.config import BaseConfig 
from dataclasses import dataclass 
from utils.common import seed_everything
from typing import * 
@dataclass
class FeedForwardConfig(BaseConfig):
    Model: BaseConfig 
    num_steps: int = 50
    cfg_strength: float = 4.0 
    type: str = 'feed-forward'
    seed: Optional[int] = None

@SYSTEM_REGISTRY.register('feed-forward')
class FeedForwardSystem:
    Config = FeedForwardConfig

    def __init__(self, cfg: FeedForwardConfig, model=None):
        self.cfg = cfg 
        if self.cfg.seed is not None:
            seed_everything(self.cfg.seed)
        self.model = model or instantiate_model(cfg.Model)
    
    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self

    @torch.no_grad()
    def __call__(
        self,
        source: torch.Tensor | List[torch.Tensor],
        prompt: str | List[str],
        negative_prompt: str | List[str] = "",
        height: int = None,
        width: int = None,
        **kwargs
    ):
        if not isinstance(source, list):
            source = [source]
        source = [s.to(self.model.device, self.model.dtype) for s in source]
        input_height, input_width = source[0].shape[-2:]
        if height is None:
            height = input_height
        if width is None:
            width = input_width
        return self.model.sample(
            prompt=prompt, 
            image=source,
            negative_prompt=negative_prompt,
            num_steps=self.cfg.num_steps,
            cfg_strength=self.cfg.cfg_strength,
            height=height,
            width=width,
            **kwargs
        )