"""
Implementation of Score Disillation Sampling for optimizing 3D representations 
Please cite the original paper "DreamFusion: Text-to-3D using 2D Diffusion" (ICLR 2023)
if you find this algorithm useful 

This is a simple, abstract, straightforward implementation, for a comprehensive implementation, refer to threestudio (https://github.com/threestudio-project/threestudio)
"""
from utils.registry import SYSTEM_REGISTRY
from utils.factory import (
    instantiate_renderer,
    instantiate_guidance,
    instantiate_model
)
from utils.config import BaseConfig 
from dataclasses import dataclass 
from system.utils.prompt_processor import PromptProcessor, PromptProcessorConfig
from utils.common import seed_everything
from typing import * 
import imageio 
import torch 
import os 
import random 
import numpy as np 
from torchvision import transforms 
from tqdm import tqdm 
from utils.logger import setup_logging, get_logger
setup_logging()
log = get_logger(__name__)
@dataclass
class ScoreDistillationConfig(BaseConfig):
    min_step_percent: float = 0.02 
    max_step_percent: float = 0.98 
    true_cfg_strength: float = 4.0 
    time_schedule: str = 'linear' # Choose from ['random', 'linear']
    weighting_strategy: str = 'inverse_linear' # Choose from ['constant', 'linear', 'inverse_linear', 'stochastic']
    weight: float = 0.5 
    num_steps: int = 50 
    iters: int = 500 
    accumulation_steps: int = 6
    batch_size: int = 1 
    # Sampled range of camera 
    min_azi: float = -180. 
    max_azi: float = 180. 
    min_ele: float = -10. 
    max_ele: float = 80. 
    min_fov: float = 49.1 
    max_fov: float = 49.1 
    min_radi: float = 1.7 
    max_radi: float = 1.7 
    invert_bg_prob: float = 0.5 
    res: int | List[int] = 512 
    res_steps: int | List[int] = 0 # Use list for annealing training resolution
    verbose: bool = False # Render video every 100 steps 
    seed: Optional[int] = None 
    Guidance: BaseConfig = None 
    Renderer: BaseConfig = None 
    PromptProcessor: BaseConfig = None 
    RFModel: BaseConfig = None 
    extract_mesh: bool = True
    type: str = 'score-distillation'

@SYSTEM_REGISTRY.register('score-distillation')
class ScoreDistillationSystem:
    Config = ScoreDistillationConfig 
    def __init__(self, cfg: ScoreDistillationConfig, device='cuda'):
        self.cfg = cfg 
        self.device = device 
        self.renderer = instantiate_renderer(cfg.Renderer).to(device)
        self.guidance = instantiate_guidance(cfg.Guidance)
        self.model = instantiate_model(cfg.RFModel).to(device)
        if self.cfg.seed is not None:
            seed_everything(self.cfg.seed)
        self.min_step = int(cfg.num_steps * cfg.min_step_percent)
        self.max_step = int(cfg.num_steps * cfg.max_step_percent)
        self.prompt_processor = PromptProcessor(PromptProcessorConfig(**cfg.PromptProcessor))
        self.timesteps = None 
        res = self.cfg.res if isinstance(self.cfg.res, int) else self.cfg.res[0]
        self.timesteps, _ = self.model.get_schedule(num_steps=self.cfg.num_steps, height=res, width=res)
        self.timesteps = self.timesteps.to(self.device)

    def get_res(self, step):
        if isinstance(self.cfg.res, int):
            assert self.cfg.res_steps == 0, "res_steps should be 0 when res is an int" 
            return self.cfg.res

        for res, step_threshold in zip(self.cfg.res, self.cfg.res_steps):
            if step < step_threshold:
                self.timesteps, _ = self.model.get_schedule(num_steps=self.cfg.num_steps, height=res, width=res)
                self.timesteps = self.timesteps.to(self.device)
                return res
    
    def to(self, device):
        self.device = device 
        self.renderer.to(device)    
        self.model.to(device)
        self.timesteps = self.timesteps.to(device)
        return self

    def prepare_train(self, prompts: str | Dict, checkpoint = None):
        self.step = 0
        self.renderer.initialize(checkpoint)
        self.renderer.training_setup()
        self.prompt_processor.set_prompt(prompts, self.model)
    
    def _sample_timestep(self, step_ratio, bs = 1):
        step_ratio = torch.tensor(step_ratio) 
        if self.cfg.time_schedule == 'random':
            return torch.randint(self.min_step, self.max_step + 1, (bs, ))

        elif self.cfg.time_schedule == 'square':
            # Proposed in HiFA
            indices = self.min_step + (self.max_step - self.min_step) * torch.sqrt(step_ratio)
            return indices.repeat(bs).long()
        elif self.cfg.time_schedule == 'linear':
            indices = self.min_step + (self.max_step - self.min_step) * step_ratio
            return indices.repeat(bs).long()
        else:
            raise ValueError(f'Unknown timestep schedule: {self.cfg.time_schedule}')
    
    def _get_weighting(self, t: float):
        if self.cfg.weighting_strategy == 'constant':
            return self.cfg.weight 
        elif self.cfg.weighting_strategy == 'linear':
            return self.cfg.weight *  t
        elif self.cfg.weighting_strategy == 'inverse_linear':
            return self.cfg.weight * (1 - t)
        elif self.cfg.weighting_strategy == 'stochastic':
            u = torch.normal(mean=0, std=1, size=(1,), device=self.device)
            return  torch.nn.functional.sigmoid(u)
        else:
            raise ValueError(f'Unkown weighting strategy: {self.cfg.weighting_strategy}')

    def train_step(self):
        loss = 0.0 
        self.step += 1
        indices = self._sample_timestep(self.step / self.cfg.iters, self.cfg.batch_size).to(self.device)
        t = self.timesteps[indices]
        w = self._get_weighting(t[0].item())
        res = self.get_res(self.step)
        for _ in range(self.cfg.accumulation_steps):
            renderings = []
            azimuths = []
            elevations = []
            
            for _ in range(self.cfg.batch_size): 
                azi = random.uniform(self.cfg.min_azi, self.cfg.max_azi)
                ele = random.uniform(self.cfg.min_ele, self.cfg.max_ele)
                fov = random.uniform(self.cfg.min_fov, self.cfg.max_fov)
                radi = random.uniform(self.cfg.min_radi, self.cfg.max_radi)

                bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.cfg.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device=self.device)

                out = self.renderer.render(
                    azi, ele, radi, fov, width=res, height=res,
                    bg_color=bg_color, 
                )

                azimuths.append(azi)
                elevations.append(ele)
                rendering = out["render"].unsqueeze(0)
                loss += self.renderer.get_reg(out, step=self.step)
                #NOTE: This unsqueeze is tailored for the QwenImage's video vae 
                rendering = rendering.unsqueeze(2).to(torch.bfloat16)
                renderings.append(rendering)

            cond = self.prompt_processor.get_prompt(self.model, azimuths, elevations)
            renderings = torch.cat(renderings, dim=0).to(self.device)

            guidance = self.guidance(self.model, renderings, t, cond, cfg_strength=self.cfg.true_cfg_strength)
            loss += w * guidance['loss']
        
        loss.backward()
        self.renderer.representation.update(self.step, out)

        return {
            "loss": loss.item(), 
            "azimuth": azimuths,
            "elevation": elevations,
            "timestep": t.cpu(),
            'rend': renderings.detach().cpu(),
            "velocity": guidance['velocity'],
            "latents": guidance['latents']
        } 

    def __call__(self, prompts: str | Dict, checkpoint = None, output_path = None, verbose: bool = True):
        self.prepare_train(prompts, checkpoint)
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        for _ in tqdm(range(self.cfg.iters), disable = not verbose):
            guidance = self.train_step()
            if self.cfg.verbose and (self.step % 100 == 0 or self.step == 1):
                log.info(f"[Step {self.step}]: Loss: {guidance['loss']}")
                step_path = os.path.join(output_path, f'step_{self.step:04d}')
                os.makedirs(step_path, exist_ok=True)
                video = self.renderer.render_video(r=self.cfg.max_radi, fov=self.cfg.max_fov, bg_color=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=self.device))
                imageio.mimwrite(os.path.join(step_path, f'{self.step:04d}.mp4'), video, fps=30)
                pred = guidance['latents'].to(self.device) - guidance['timestep'].to(self.device).view(-1, 1, 1) * guidance['velocity'].to(self.device)
                for a, e, t, rend, p in zip(guidance['azimuth'], guidance['elevation'], guidance['timestep'], guidance['rend'], pred):
                # Save training rendering and corresponding one-step prediction 
                    rend = transforms.ToPILImage()(rend.squeeze().cpu().float().clamp(0, 1))
                    rend.save(os.path.join(step_path, f'rend_azi{a:.1f}_ele{e:.1f}_t{t:.3f}.png'))
                
                    height, width = rend.size
                    p = self.model.decode(p.unsqueeze(0).detach(), output_type='pil', height=height, width=width)[0]
                    p.save(os.path.join(step_path, f'pred_azi{a:.1f}_ele{e:.1f}_t{t:.3f}.png'))
        self.renderer.representation.save(os.path.join(output_path, 'guidance_model.ply'))
        torch.cuda.empty_cache()
        if self.cfg.extract_mesh:
            mesh = self.renderer.extract_mesh(verbose)
            rotation_matrix = np.array([
            [-1, 0, 0, 0],
            [ 0, 0, 1, 0],
            [ 0, 1, 0, 0],
            [ 0, 0, 0, 1]
            ])
            mesh.apply_transform(rotation_matrix)
            mesh.export(os.path.join(output_path, 'guidance_mesh.glb'))

        


