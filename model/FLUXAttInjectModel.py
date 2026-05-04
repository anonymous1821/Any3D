"""
Some attention manipulation based methods, espectially attention injection methods (RFSolver-Edit, FireFlow)
requires recording attention values during inversion and applying during editing. This is a custom module 
that perform such recording and injection.

Note that the model is implemented upon the official implementation of FLUX instead of diffusers (FLUXModel),
which has been evidented that may outperform diffusers version a little. 
"""
from utils.registry import MODEL_REGISTRY
from .FLUXModel import FLUXModel, FLUXConfig, retrieve_timesteps, calculate_shift
import torch
from torch import nn, Tensor
from typing import *
from .custom.flux.flux import Flux, FluxParams
from .custom.flux.autoencoder import AutoEncoder, AutoEncoderParams
from .custom.flux.conditioner import HFEmbedder
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_sft
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
import os
import numpy as np
from utils.logger import setup_logging, get_logger
setup_logging()
log = get_logger(__name__)

@MODEL_REGISTRY.register('flux-att-inject')
class FLUXAttInjectModel(FLUXModel):
    Config = FLUXConfig
    
    def __init__(self, cfg: FLUXConfig):
        # Do not call super().__init__(cfg) to avoid loading diffusers pipeline
        self.cfg = cfg
        self.device = cfg.device
        self.dtype = torch.bfloat16
        
        # Determine params and repo based on model name
        if "schnell" in cfg.model_name:
            repo_id = "black-forest-labs/FLUX.1-schnell"
            repo_flow = "flux1-schnell.safetensors"
            guidance_embed = False
        else:
            # Default to dev
            repo_id = "black-forest-labs/FLUX.1-dev" 
            repo_flow = "flux1-dev.safetensors"
            guidance_embed = True
        
        repo_ae = "ae.safetensors"

        # Flux Params
        flux_params = FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=guidance_embed,
        )

        # AutoEncoder Params
        ae_params = AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        )

        log.info("Initializing FluxAttnInjectModel...")
        self.transformer = Flux(flux_params).to(self.device, dtype=self.dtype)
        self._load_checkpoint(self.transformer, repo_id, repo_flow)

        self.vae = AutoEncoder(ae_params).to(self.device, dtype=self.dtype)
        self._load_checkpoint(self.vae, repo_id, repo_ae)

        self.text_encoder = HFEmbedder("openai/clip-vit-large-patch14", max_length=77, is_clip=True, torch_dtype=self.dtype).to(self.device)

        self.text_encoder_2 = HFEmbedder("google/t5-v1_1-xxl", max_length=512, is_clip=False, torch_dtype=self.dtype).to(self.device)

        # Setup scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")

        # Setup attributes
        self.latents_mean = self.vae.shift_factor 
        self.latents_std = self.vae.scale_factor
        self.vae_scale_factor = 8 # 2**(len(ch_mult)-1) = 2**(4-1) = 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        
        self.tokenizer_max_length = 77
        self.max_sequence_length = 512
        self.num_channels_latents = self.transformer.params.in_channels // 4 

    def _load_checkpoint(self, model, repo_id, filename):
        try:
            ckpt_path = hf_hub_download(repo_id, filename)
            log.info(f"Loading weights from {ckpt_path}")
            sd = load_sft(ckpt_path, device=str(self.device))
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if len(missing) > 0:
                log.warning(f"Missing keys: {len(missing)}")
            if len(unexpected) > 0:
                log.warning(f"Unexpected keys: {len(unexpected)}")
        except Exception as e:
            log.error(f"Failed to load weights for {filename}: {e}")

    def to(self, device):
        self.device = device
        self.transformer = self.transformer.to(device)
        self.text_encoder = self.text_encoder.to(device)
        self.text_encoder_2 = self.text_encoder_2.to(device)
        self.vae = self.vae.to(device)
        return self

    def _get_t5_prompt_embeds(self, prompt: Union[str, List[str]] = None):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        return self.text_encoder_2(prompt)

    def _get_clip_prompt_embeds(self, prompt: Union[str, List[str]]):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        return self.text_encoder(prompt)

    def encode(self, x: torch.Tensor, generator: Optional[torch.Generator] = None):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device, self.dtype)
        bs = x.shape[0]
        x = x * 2. - 1.
        # Custom AE encode returns z directly (already sampled/meaned/scaled)
        latents = self.vae.encode(x)
        # latents = (latents - self.latents_mean) * self.latents_std
        
        latent_height, latent_width = latents.shape[-2:]
        latents = self._pack_latents(latents, bs, self.num_channels_latents, latent_height, latent_width)
        return latents

    def decode(self, latents: torch.Tensor, height: int, width: int, output_type: str = 'pt'):
        if len(latents.shape) == 3:
            latents = self._unpack_latents(latents, width, height, self.vae_scale_factor)
        # latents = (latents / self.latents_std) + self.latents_mean
        # Custom AE decode returns tensor directly
        x = self.vae.decode(latents)
        x = self.image_processor.postprocess(x, output_type = output_type)
        return x 


    def get_v_prediction(
        self,
        x_t: torch.Tensor, 
        t: float, 
        cfg_strength: float = 1.0, 
        guidance_scale: float = 3.5, 
        width: int = None, 
        height: int = None,
        latent_image_ids = None,
        cond: Dict = None,
        pos_cond: Dict = None, 
        info: Dict = None,
        **kwargs
    ):
        do_cfg = cfg_strength > 1. 
        bs = x_t.shape[0]
        t = torch.full((bs,), t, device=self.device, dtype=self.dtype)
        
        # Use params.guidance_embed from the custom Flux model
        if self.transformer.params.guidance_embed:
            guidance = torch.full((bs, ), guidance_scale, device=self.device, dtype=self.dtype)
        else:
            guidance = None
            
        if not latent_image_ids:
            assert (width and height)
            height = 2 * (int(height) // (self.vae_scale_factor * 2))
            width = 2 * (int(width) // (self.vae_scale_factor * 2))
            latent_image_ids = self._get_latent_image_ids(bs, height // 2, width // 2, self.device, torch.bfloat16)
        
        if info is None:
            info = {}

        # Add 't' to info if not present, might be useful for injection naming
        if 't' not in info:
            info['t'] = t[0].item() # Assuming t is constant across batch for this use case

        if do_cfg:
            cond = self.cond if cond is None else cond
            
            v_pred_out, _ = self.transformer(
                hidden_states=x_t,
                img_ids=latent_image_ids,
                encoder_hidden_states=cond["encoder_hidden_states"],
                txt_ids=cond["txt_ids"],
                timestep=t,
                pooled_projections=cond["pooled_projections"],
                guidance=guidance,
                info=info,
                return_dict=False
            )
            v_pred, neg_v_pred = v_pred_out.chunk(2)
            v_pred = neg_v_pred + cfg_strength * (v_pred - neg_v_pred)
        else:
            cond = self.pos_cond if pos_cond is None else pos_cond
            v_pred, _ = self.transformer(
                hidden_states=x_t,
                img_ids=latent_image_ids,
                encoder_hidden_states=cond["encoder_hidden_states"],
                txt_ids=cond["txt_ids"],
                timestep=t,
                pooled_projections=cond["pooled_projections"],
                guidance=guidance,
                info=info,
                return_dict=False
            )

        return v_pred, info

