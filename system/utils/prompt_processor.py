"""
Prompt Processor, used for ScoreDistillationSystem for view-dependent prompting

For multi-view prompts, expected input is of dict:
{
    "prompt_front": "...",
    "prompt_left": "...",
    "prompt_right": "...", 
    "prompt_back": "...", 
    "prompt_overhead": "...",
    "prompt_negative": "..."
}

Additionally, input can be 8-view by adding keys:
"prompt_front-left", "prompt_front-right", "prompt_back-left", "prompt_back-right"
"""
import torch

from utils.config import BaseConfig 
from dataclasses import dataclass, field
from typing import *
from utils.logger import setup_logging, get_logger
setup_logging()
log = get_logger(__name__)

def shift_azimuth_deg(azimuth):
    return (azimuth + 180) % 360 - 180

@dataclass
class PromptProcessorConfig(BaseConfig):

    # directions maps view name -> (start_deg, end_deg) in degrees.
    # Ranges are in the [-180, 180] azimuth space and may wrap (start > end).
    directions: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "front": (-45.0, 45.0),
            "left": (45.0, 135.0),
            "back": (135.0, -135.0),
            "right": (-135.0, -45.0),
        }
    )

    overhead_thr: float = 50.0
    view_dependent_prompting: bool = True 

class PromptProcessor:
    Config = PromptProcessorConfig
    def __init__(self, cfg: PromptProcessorConfig):
        self.cfg = cfg
        self.directions = cfg.directions
        self.overhead_thr = cfg.overhead_thr
    
    @torch.no_grad()
    def set_prompt(self, prompts, model):
        # negative_prompt = prompts.get("prompt_negative", "")
        negative_prompt = prompts.get("negative_prompt", "")
        prompts = {k: v for k, v in prompts.items() if k != "prompt_negative"}
        self.cond = {}
        for key, prompt in prompts.items():
            if key == 'prompt':
                self.cond['raw'], _ = model.set_prompt(prompt, negative_prompt)
            else:
                name = key.split("_")[-1]
                self.cond[name], _ = model.set_prompt(prompt, negative_prompt)
    
    @torch.no_grad()
    def get_prompt(self, model, azimuths, elevations):
        bs = len(azimuths)
        if not self.cfg.view_dependent_prompting:
            return model.batch_conds([self.conda['raw']] * bs)
        conds = []
        for a, e in zip(azimuths, elevations):
            if e > self.overhead_thr:
                conds.append(self.cond["overhead"])
                continue
            for direction, (s, e) in self.directions.items():
                if s <= e:
                    in_range = (a >= s) and (a < e)
                else:
                    # wrapped range (e.g. start=135, end=-135) covers [s,180] U [-180,e)
                    in_range = (a >= s) or (a < e)
                if in_range:
                    conds.append(self.cond[direction])
                    break
        return model.batch_conds(conds)