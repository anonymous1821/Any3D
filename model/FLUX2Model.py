"""
Implementation of FLUX2 model family (https://github.com/black-forest-labs/flux2)
"""
from utils.registry import MODEL_REGISTRY
from utils.config import BaseConfig
from dataclasses import dataclass
from .RFModel import RFModel
import torch
import numpy as np 
from typing import * 
from einops import rearrange
from .flux2.util import load_flow_model, load_text_encoder, load_ae, FLUX2_MODEL_INFO
from .flux2.sampling import get_schedule, batched_prc_img, batched_prc_txt, denoise, prc_img, prc_txt, encode_image_refs, listed_prc_img
from PIL import Image

@dataclass
class FLUX2ModelConfig(BaseConfig):
    # Available models:
    # - flux.2-klein-4b
    # - flux.2-klein-9b
    # - flux.2-klein-base-4b
    # - flux.2-klein-base-9b
    # - flux.2-dev
    model_name: str = "flux.2-klein-9b"
    device: str = 'cuda:0'
    type: str = 'flux2'

@MODEL_REGISTRY.register('flux2')
class FLUX2Model(RFModel):
    Config = FLUX2ModelConfig

    def __init__(self, cfg: FLUX2ModelConfig):
        self.cfg = cfg
        self.model_name = cfg.model_name.lower()
        self.device = device = cfg.device
        self.dtype = torch.bfloat16
        
        # Load models
        self.transformer = load_flow_model(self.model_name, device=device)
        self.ae = load_ae(self.model_name, device=device)
        self.text_encoder = load_text_encoder(self.model_name, device=device)
        
        self.num_channels_latents = 128
        self.vae_scale_factor = 16 
        
        self.transformer.eval()
        self.ae.eval()
        
        # Freeze models
        self.transformer.requires_grad_(False)
        self.ae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Get defaults from config
        self.default_steps = FLUX2_MODEL_INFO[self.model_name]["defaults"]["num_steps"]
        self.default_guidance = FLUX2_MODEL_INFO[self.model_name]["defaults"]["guidance"]
        self.guidance_distilled = FLUX2_MODEL_INFO[self.model_name]["guidance_distilled"]
        # Cache for conditions
        self.cond = None
        self.pos_cond = None
        self.ref_context = None

    def to(self, device):
        self.transformer = self.transformer.to(device)
        self.ae = self.ae.to(device)
        self.text_encoder = self.text_encoder.to(device)
        self.device = device
        return self
    
    def get_latents_mask(
        self,
        mask,
        height, 
        width
    ):
        if isinstance(mask, list):
            mask = torch.stack(mask, dim=0)
        mask = mask.to(self.device, self.dtype)
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        mask = torch.nn.functional.interpolate(mask, size=(latent_height, latent_width))[:, :1, :, :]
        mask = mask.repeat(1, self.num_channels_latents, 1, 1)
        mask = rearrange(mask, "b c h w -> b (h w) c")
        return mask

    def encode(
        self, 
        x: torch.Tensor | List[torch.Tensor],
        generator: Optional[torch.Generator] = None
    ):
        """
        x (torch.Tensor): shape (B, 3, H, W) or (3, H, W), within range [0, 1]
        Returns packed latents: (B, L, C)
        """
        if isinstance(x, list):
            x = torch.stack(x, dim=0)
        x = x.float().to(self.device)
        bs = x.shape[0]
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        # Normalize to [-1, 1]
        x = x * 2. - 1.
        
        # AE Encode
        latents = self.ae.encode(x) # (B, 128, H/16, W/16)
        
        # Pack latents using sampling.prc_img logic
        # batched_prc_img expects tensor input (B, C, H, W)
        packed_latents, x_ids = batched_prc_img(latents)
        
        return packed_latents

    @staticmethod
    def _pack_latents(
        latents: torch.Tensor,
        num_channels_latents: int,
        height: int, 
        width: int,
        **kwargs
    ):
        latent_height = height // 16
        latent_width = width // 16  
        return rearrange(latents, "b c h w -> b (h w) c", c=num_channels_latents, h=latent_height, w=latent_width) 
    
    @staticmethod
    def _unpack_latents(
        latents: torch.Tensor,
        num_channels_latents: torch.Tensor,
        height: int, 
        width: int,
        **kwargs
    ):
        latent_height = height // 16
        latent_width = width // 16  
        return rearrange(latents, "b (h w) c -> b c h w", c=num_channels_latents, h=latent_height, w=latent_width)

    def decode(
        self, 
        latents: torch.Tensor,
        height: int,
        width: int,
        output_type: str = 'pt'
    ):
        """
        latents: (B, L, C)
        """
        bs, l, c = latents.shape
        
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        
        # Unpack latents
        # rearrange (B, (h w), c) -> (B, c, h, w)
        latents = rearrange(latents, "b (h w) c -> b c h w", h=latent_h, w=latent_w)
        
        # Decode
        x = self.ae.decode(latents) # (B, 3, H, W)
        
        # Postprocess
        x = (x + 1.0) / 2.0
        x = torch.clamp(x, 0.0, 1.0)
        
        if output_type == 'pil':
            x = x.permute(0, 2, 3, 1).float().cpu().numpy()
            x = (x * 255).round().astype(np.uint8)
            images = [Image.fromarray(img) for img in x]
            return images
            
        return x 

    def set_reference_image(self, image):
        # Prepare reference images
        ref_tokens = None
        ref_ids = None
        if image is not None:
             # Check if PIL
             is_pil = False
             if isinstance(image, Image.Image):
                 is_pil = True
             elif isinstance(image, list) and len(image) > 0 and isinstance(image[0], Image.Image):
                 is_pil = True
                 
             if is_pil:
                 if isinstance(image, Image.Image):
                     image = [image]
                 ref_tokens, ref_ids = encode_image_refs(self.ae, image)
             else:
                 ref_tokens, ref_ids = self._encode_tensor_refs(image)
        
        self.ref_context = {
            'ref_tokens': ref_tokens,
            'ref_ids': ref_ids
        }
        
        return self.ref_context

    def set_prompt(
        self, 
        prompt,
        negative_prompt = "",
    ):
        if isinstance(prompt, str):
            prompt = [prompt]
        
        bs = len(prompt)
        
        if not self.guidance_distilled:
            # For non-distilled models (Base), we use CFG which requires negative prompt
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * bs
            prompts = prompt + negative_prompt
        else:
            # Distilled models use guidance embedding, negative prompt is ignored
            prompts = prompt
        
        # Text encoding
        embeddings = self.text_encoder(prompts)
        embeddings = embeddings.to(self.device, self.dtype)
        
        # Process text IDs
        txt_latents, txt_ids = batched_prc_txt(embeddings)
        
        self.cond = {
            'ctx': txt_latents,
            'ctx_ids': txt_ids,
        }
        
        if not self.guidance_distilled:
            self.pos_cond = {
                'ctx': txt_latents[:bs],
                'ctx_ids': txt_ids[:bs],
            }
        else:
            self.pos_cond = self.cond
        
        return self.cond, self.pos_cond 

    def get_v_prediction(
        self,
        latents: torch.Tensor,  
        t: torch.Tensor,
        cfg_strength: float, 
        width: int, 
        height: int,
        img_ids: torch.Tensor = None,
        cond: Dict = None,
        ref_tokens: torch.Tensor = None,
        ref_ids: torch.Tensor = None,
        **kwargs
    ):
        bs = latents.shape[0]
        
        if isinstance(t, float) or t.ndim == 0:
            t = torch.full((bs,), t, device=self.device, dtype=self.dtype)
        
        if cond is None:
            cond = self.cond
        
        if ref_tokens is None:
            ref_tokens = self.ref_context.get('ref_tokens') if self.ref_context else None
        if ref_ids is None:
            ref_ids = self.ref_context.get('ref_ids') if self.ref_context else None
        
        if img_ids is None:
            # Reconstruct img_ids if not provided
            latent_h = height // self.vae_scale_factor
            latent_w = width // self.vae_scale_factor
            
            dummy_latents = torch.zeros((bs, 128, latent_h, latent_w), device=self.device)
            _, img_ids = batched_prc_img(dummy_latents)
            
        # Handle reference images
        original_seq_len = latents.shape[1]
        
        if ref_tokens is not None and ref_ids is not None:
            # ref_tokens: (B, L_ref, C) or (1, L_ref, C)
            if ref_tokens.shape[0] != bs:
                ref_tokens = ref_tokens.repeat(bs, 1, 1)
                ref_ids = ref_ids.repeat(bs, 1, 1)
            
            latents = torch.cat([latents, ref_tokens], dim=1)
            img_ids = torch.cat([img_ids, ref_ids], dim=1)

        if self.guidance_distilled:
            # Distilled models use guidance embedding
            guidance_vec = torch.full((bs,), cfg_strength, device=self.device, dtype=self.dtype)
            
            pred = self.transformer(
                x=latents.to(self.dtype),
                x_ids=img_ids,
                timesteps=t,
                guidance=guidance_vec,
                **cond
            )
            
            # Slice output to remove reference tokens if present
            if ref_tokens is not None:
                pred = pred[:, :original_seq_len, :]
                
            return pred
        else:
            # Non-distilled models use CFG
            do_cfg = cfg_strength > 1.0
            
            if do_cfg:
                # Duplicate inputs for CFG
                latents_in = torch.cat([latents] * 2, dim=0)
                img_ids_in = torch.cat([img_ids] * 2, dim=0)
                t_in = torch.cat([t] * 2, dim=0)
                
                # Use combined cond (pos + neg)
                if cond is None or (cond['ctx'].shape[0] != bs * 2):
                    # Fallback if cond not prepared correctly or passed explicitly
                     cond_in = self.cond
                else:
                     cond_in = cond
                
                # Guidance vector is ignored by model but required by signature
                guidance_vec = torch.zeros((bs * 2,), device=self.device, dtype=self.dtype)
                
                pred_out = self.transformer(
                    x=latents_in.to(self.dtype),
                    x_ids=img_ids_in,
                    timesteps=t_in,
                    guidance=guidance_vec,
                    **cond_in
                )
                
                # Slice output to remove reference tokens if present
                if ref_tokens is not None:
                    pred_out = pred_out[:, :original_seq_len, :]
                
                pred_pos, pred_neg = pred_out.chunk(2)
                pred = pred_neg + cfg_strength * (pred_pos - pred_neg)
                return pred
            else:
                # Unconditional (or just positive if cfg=1)
                # If cfg=1, we just run positive. 
                # Ensure we use pos_cond
                if cond is None or (cond['ctx'].shape[0] != bs):
                     # If cond is combined, use pos_cond
                     # But we need to make sure pos_cond is available
                     if self.pos_cond is not None:
                         cond = self.pos_cond
                     else:
                         # Fallback: slice cond
                         # Need to make a new dict to avoid modifying original
                         cond = {
                             'ctx': cond['ctx'][:bs],
                             'ctx_ids': cond['ctx_ids'][:bs],
                         }
                
                guidance_vec = torch.zeros((bs,), device=self.device, dtype=self.dtype)
                
                pred = self.transformer(
                    x=latents.to(self.dtype),
                    x_ids=img_ids,
                    timesteps=t,
                    guidance=guidance_vec,
                    **cond
                )
                
                # Slice output to remove reference tokens if present
                if ref_tokens is not None:
                    pred = pred[:, :original_seq_len, :]
                    
                return pred

    def get_schedule(
        self,
        num_steps: int,
        width: int, 
        height: int, 
        **kwargs
    ):
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        image_seq_len = latent_h * latent_w
        
        timesteps = get_schedule(num_steps, image_seq_len)
        return timesteps, timesteps

    def _encode_tensor_refs(self, ref_tensors: List[torch.Tensor] | torch.Tensor):
        if isinstance(ref_tensors, torch.Tensor):
            if ref_tensors.ndim == 3:
                ref_tensors = [ref_tensors]
            elif ref_tensors.ndim == 4:
                ref_tensors = [t for t in ref_tensors]
            else:
                raise ValueError(f"Invalid shape for ref_tensors: {ref_tensors.shape}")
        
        encoded_refs = []
        for x in ref_tensors:
            if x.ndim == 3:
                x = x.unsqueeze(0)
            
            # Assume input is [0, 1], normalize to [-1, 1]
            x = x * 2. - 1.
            x = x.float().to(self.device)
            
            # Encode
            encoded = self.ae.encode(x)[0] 
            encoded_refs.append(encoded)
            
        scale = 10
        t_off = [scale + scale * t for t in torch.arange(0, len(encoded_refs))]
        t_off = [t.view(-1) for t in t_off]
        
        ref_tokens, ref_ids = listed_prc_img(encoded_refs, t_coord=t_off)
        
        ref_tokens = torch.cat(ref_tokens, dim=0)
        ref_ids = torch.cat(ref_ids, dim=0)
        
        ref_tokens = ref_tokens.unsqueeze(0)
        ref_ids = ref_ids.unsqueeze(0)
        
        return ref_tokens.to(dtype=self.dtype), ref_ids

    @torch.no_grad()
    def sample(
        self,
        prompt: str | List[str],
        image: torch.Tensor | List[torch.Tensor] = None,
        negative_prompt: str = "",
        num_steps: int = None, 
        cfg_strength: float = None,
        height: int = 1024, 
        width: int = 1024,
        generator: torch.Generator = None,
        start: int = 0, 
        latents: torch.Tensor | List[torch.Tensor] = None, 
        mask: torch.Tensor | List[torch.Tensor] = None, # 1 for editied region, 0 for static region
        inpaint_latents: torch.Tensor | List[torch.Tensor] | List[List[torch.Tensor]]= None, 
        **kwargs
    ):
        if num_steps is None:
            num_steps = self.default_steps
        if cfg_strength is None:
            cfg_strength = self.default_guidance
        bs = len(prompt) if isinstance(prompt, list) else 1
        self.set_prompt(prompt, negative_prompt)
        self.set_reference_image(image)
        
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        
        # Prepare latents
        if latents is None:
            z = torch.randn(
                (bs, 128, latent_h, latent_w),
                device=self.device,
                dtype=self.dtype,
                generator=generator
            )
            latents, img_ids = batched_prc_img(z)
        else:
            if len(latents.shape) > 3: # (B, C, H, W)
                latents, img_ids = batched_prc_img(latents)
            else:
                # Already packed? We need img_ids.
                dummy = torch.zeros((bs, 128, latent_h, latent_w), device=self.device)
                _, img_ids = batched_prc_img(dummy)

        timesteps, _ = self.get_schedule(num_steps, width, height)
        
        # Prepare inpainting
        inpaint_noise = None
        
        if mask is not None:
            mask = [mask] if not isinstance(mask, list) else mask
            mask = self.get_latents_mask(mask, height, width)
            if isinstance(inpaint_latents, torch.Tensor):
                inpaint_noise = torch.randn_like(inpaint_latents)
            else:
                latents = latents * mask + inpaint_latents[0].to(latents.device, latents.dtype) * (1 - mask)

        # Sampling Loop
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[start:-1], timesteps[start+1:])):

            v_pred = self.get_v_prediction(
                latents,
                t_curr, 
                cfg_strength,
                width,
                height,
                img_ids=img_ids,
                cond=self.cond
            )
            
            latents = latents + (t_prev - t_curr) * v_pred
            
            if mask is not None and isinstance(inpaint_latents, torch.Tensor):
                noisy_inpaint = t_prev * inpaint_noise + (1 - t_prev) * inpaint_latents
                latents = latents * mask + noisy_inpaint * (1 - mask)
                
            elif mask is not None and isinstance(inpaint_latents, list):
                curr_inpaint = inpaint_latents[i+1]
                if len(curr_inpaint.shape) > 3:
                     curr_inpaint, _ = batched_prc_img(curr_inpaint)
                latents = latents * mask + curr_inpaint.to(latents.device, latents.dtype) * (1 - mask)

        return self.decode(latents, height, width, output_type='pil')