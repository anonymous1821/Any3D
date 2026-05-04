import torch 
from model.RFModel import RFModel 
from dataclasses import dataclass 
from torch import Tensor 
from typing import List

@dataclass 
class InversionOption:
    start_timestep: float 
    stop_timestep: float 
    num_steps: int
    num_inversion_steps: int
    guidance_scale: float 

class Inversion():
    def __init__(self, config):
        pass 
    
    def align_bs(
        self,
        sample: Tensor, 
        prompt: str | List[str]
    ):
        if isinstance(prompt, list):
            assert (sample.shape[0] == 1) or (len(prompt) == 1) or (sample.shape[0] == len(prompt))
            bs = max(sample.shape[0], len(prompt))
            sample = sample.expand(bs, -1, -1, -1) if sample.shape[0] == 1 else sample
            prompt = prompt * bs if len(prompt) == 1 else prompt
        else:
            prompt = [prompt] * sample.shape[0]
            bs = 1
        return sample, prompt, bs

    def to(self, device: str):
        self.device = device
        return self  

    def invert(
        self,
        sample, 
        prompt: str | List[str],
        model: RFModel
    ):
        pass

    def edit(
        self,
        sample: Tensor, 
        prompt: str | List[str],
        model: RFModel
    ):
        pass

    def __call__(self):
        pass 

