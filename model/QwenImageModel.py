"""
Implementation of QwenImage model (https://github.com/QwenLM/Qwen-Image)
Please cite the original paper "Qwen-Image Technical Report" (arXiv 2025)
if you find this model helpful
"""
from torch.nn.utils.rnn import pad_sequence
from diffusers import DiffusionPipeline 
from diffusers.image_processor import VaeImageProcessor 
from utils.registry import MODEL_REGISTRY
from utils.config import BaseConfig
from dataclasses import dataclass
import inspect 
from .RFModel import RFModel
import torch
import numpy as np 
from typing import * 
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents 
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

@dataclass
class QwenImageModelConfig(BaseConfig):
    model_name: str
    device: str='cuda:0'

@MODEL_REGISTRY.register('qwen-image')
class QwenImageModel(RFModel):
    Config = QwenImageModelConfig

    def __init__(self, cfg: QwenImageModelConfig):
        self.cfg = cfg
        model_name = cfg.model_name
        self.device = device = cfg.device
        self.dtype = torch.bfloat16
        pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=self.dtype)    
        pipe.to(device)
        self.vae = pipe.vae 
        self.transformer = pipe.transformer 
        self.tokenizer = pipe.tokenizer 
        self.text_encoder = pipe.text_encoder 
        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample)
        self.scheduler = pipe.scheduler
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        del pipe

        self.latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(device, self.dtype)
        )
        self.latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            device, self.dtype
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = 1024
        self.max_sequence_length = 512
        self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34

        self.num_channels_latents = self.transformer.config.in_channels // 4 

    def to(self, device):
        self.vae = self.vae.to(device)
        self.text_encoder = self.text_encoder.to(device)
        self.transformer = self.transformer.to(device)
        self.latents_mean = self.latents_mean.to(device)
        self.latents_std = self.latents_std.to(device)
        self.device = device
        return self
        
    
    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    def _get_qwen_prompt_embeds(
        self,
        prompt: str | List[str]
    ):

        device = self.text_encoder.device 
        detype = torch.bfloat16

        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]
        txt_tokens = self.tokenizer(
            txt, max_length=self.tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        encoder_hidden_states = self.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=device)

        return prompt_embeds, encoder_attention_mask

    def get_latents_mask(
        self,
        mask, 
        batch_size,
        height,
        width,
    ):
        if isinstance(mask, list):
            mask = torch.stack(mask, dim=0)
        latent_height = 2 * (int(height) // (self.vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (self.vae_scale_factor * 2))
        mask = torch.nn.functional.interpolate(mask, size=(latent_height, latent_width))
        mask = mask.to(self.device, self.dtype)
        mask = mask[:, 0, :, :]
        mask = self._pack_latents(
            mask.repeat(1, self.num_channels_latents, 1, 1), 
            batch_size, 
            self.num_channels_latents, 
            latent_height, 
            latent_width,
        )

        return mask

    def encode(
        self, 
        x: torch.Tensor | List[torch.Tensor],
        generator: Optional[torch.Generator] = None
    ):
        """
        x (torch.Tensor): shape (B, 3, H, W) or (3, H, W), within range [0, 1]
        """
        if isinstance(x, list):
            x = torch.stack(x, dim=0)
        x = x.to(self.vae.device, self.vae.dtype)
        bs = x.shape[0]
        if len(x.shape) == 3:
            x = x.unsqueeze(0).unsqueeze(2)
        if len(x.shape) == 4:
            x = x.unsqueeze(2)
        x = x * 2. - 1.
        x = x.to(torch.bfloat16)
        latents = retrieve_latents(self.vae.encode(x), generator = generator)
        
        latents = (latents - self.latents_mean) / self.latents_std

        latent_height, latent_width = latents.shape[-2:]
        latents = self._pack_latents(latents, bs, self.num_channels_latents, latent_height, latent_width)

        return latents 

    def decode(
        self, 
        latents: torch.Tensor,
        height: int,
        width: int,
        output_type: str = 'pt'
    ):
        if len(latents.shape) == 3:
            latents = self._unpack_latents(latents, width, height, self.vae_scale_factor)
        latents = latents * self.latents_std + self.latents_mean
        x = self.vae.decode(latents.to(torch.bfloat16), return_dict=False)[0][:, :, 0]
        x = self.image_processor.postprocess(x, output_type = output_type)
        return x 

    def set_prompt(
        self, 
        prompt,
        negative_prompt = ""
    ):
        """
        Always assume a negative prompt, disabled by setting cfg scale less than 1
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * len(prompt)
        bs = len(prompt)
        prompts = prompt + negative_prompt 
        prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompts)
        prompt_embeds = prompt_embeds[:, :self.max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :self.max_sequence_length]

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs * 2, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.view(bs * 2, seq_len)

        self.cond = {
            'encoder_hidden_states': prompt_embeds, 
            'encoder_hidden_states_mask': prompt_embeds_mask,
            'txt_seq_lens': prompt_embeds_mask.sum(dim=1).tolist()
        }

        pos_embeds = prompt_embeds.chunk(2)[0]
        pos_embeds_mask = prompt_embeds_mask.chunk(2)[0]
        self.pos_cond = {
            'encoder_hidden_states': pos_embeds,
            'encoder_hidden_states_mask': pos_embeds_mask,
            'txt_seq_lens': pos_embeds_mask.sum(dim=1).tolist()
        }

        return self.cond, self.pos_cond
    
    @torch.no_grad()
    def batch_conds(self, conds: List[Dict]): 
        """
        Integrate conditions for batch inference 
        This should incorporate zero padding to ensure the embeddings are of same shape

        We assume the input conds already incoporated a negative prompt embeddings
        We should rearrange so that all positive come first, then all negative come
        """
        encoder_hidden_states = [c['encoder_hidden_states'][0] for c in conds]
        encoder_hidden_states_mask = [c['encoder_hidden_states_mask'][0] for c in conds]
        neg_encoder_hidden_states = [c['encoder_hidden_states'][1] for c in conds]
        neg_encoder_hidden_states_mask = [c['encoder_hidden_states_mask'][1] for c in conds]
        encoder_hidden_states = encoder_hidden_states + neg_encoder_hidden_states
        encoder_hidden_states_mask = encoder_hidden_states_mask + neg_encoder_hidden_states_mask
        encoder_hidden_states = pad_sequence(encoder_hidden_states, batch_first=True, padding_value=0)
        encoder_hidden_states_mask = pad_sequence(encoder_hidden_states_mask, batch_first=True, padding_value=0)
        return {
            'encoder_hidden_states': encoder_hidden_states,
            'encoder_hidden_states_mask': encoder_hidden_states_mask,
            'txt_seq_lens': encoder_hidden_states_mask.sum(dim=1).tolist()
        }

    def get_v_prediction(
        self,
        latents: torch.Tensor,  
        t: torch.Tensor,
        cfg_strength: float,
        width: int, 
        height: int,
        img_shapes: List[List[Tuple[int, int, int]]] = None,
        cond: Dict = None,
        pos_cond: Dict = None, 
        norm_cfg: bool = True,
        **kwargs
    ):

        if img_shapes is None:
            assert width is not None and height is not None
            img_shapes = [[(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)]]
        do_cfg = cfg_strength > 1. 
        bs = latents.shape[0]
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.shape[0] == 1:
            t = t.repeat(bs)
        t = t.to(self.device, self.dtype)
        if do_cfg:
            cond = self.cond if cond is None else cond
            v_pred, neg_v_pred=self.transformer(
                hidden_states=torch.cat([latents] * 2, dim=0).to(self.dtype),
                timestep=torch.cat([t] * 2, dim=0),
                **cond,
                img_shapes=img_shapes * bs * 2,
                return_dict=False
            )[0].chunk(2)
            if norm_cfg:
                comb_pred = neg_v_pred + cfg_strength * (v_pred - neg_v_pred)
                cond_norm = torch.norm(v_pred, dim=-1, keepdim=True)
                comb_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                v_pred = comb_pred * (cond_norm / comb_norm)
            else:
                v_pred = neg_v_pred + cfg_strength * (v_pred - neg_v_pred)
        else:
            cond = self.pos_cond if pos_cond is None else pos_cond
            v_pred = self.transformer(
                hidden_states = latents.to(self.dtype),
                timestep = t,
                **cond,
                img_shapes = img_shapes * bs,
                return_dict = False
            )[0]
        
        return v_pred

    def get_schedule(
        self,
        num_steps: int,
        width: int, 
        height: int, 
        **kwargs
    ):
        latent_height = 2 * (int(height) // (self.vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (self.vae_scale_factor * 2))
        image_seq_len = (latent_height // 2) * (latent_width // 2)

        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        mu = calculate_shift(
            image_seq_len, 
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15)
        )

        timesteps, _ = retrieve_timesteps(
            self.scheduler, 
            num_steps,
            self.device, 
            sigmas = sigmas, 
            mu=mu
        )

        timesteps /= 1000 
        # NOTE: Add 0
        timesteps = torch.cat([timesteps, torch.tensor([0.], device=timesteps.device)])
        sigmas = self.scheduler.sigmas.to(self.device).float()

        return timesteps, sigmas

    @torch.no_grad()
    def sample(
        self,
        prompt: str | List[str],
        negative_prompt: str = "",
        num_steps: int = 50, 
        cfg_strength: float = 4.5,
        height: int = 1024, 
        width: int = 1024,
        generator: torch.Generator = None,
        start: int = 0, 
        latents: torch.Tensor | List[torch.Tensor] = None, 
        mask: torch.Tensor | List[torch.Tensor] = None, # 1 for editied region, 0 for static region
        inpaint_latents: torch.Tensor | List[torch.Tensor] | List[List[torch.Tensor]]= None, 
        **kwargs
    ):
        bs = len(prompt) if isinstance(prompt, list) else 1
        self.set_prompt(prompt, negative_prompt)
        num_images = len(prompt) if isinstance(prompt, list) else 1
        latent_height = 2 * (int(height) // (self.vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (self.vae_scale_factor * 2))
        if latents is None:
            latents = torch.randn(
                num_images, 
                self.num_channels_latents, 
                latent_height, 
                latent_width,
                device=self.device, 
                dtype=self.dtype,
                generator=generator
            ) 
        if len(latents.shape) > 3:
            latents = self._pack_latents(latents, num_images, self.num_channels_latents, latent_height, latent_width)
        timesteps, _ = self.get_schedule(num_steps, width, height)
        inpaint_noise = None
        if mask is not None:
            mask = [mask] if not isinstance(mask, list) else mask
            mask = self.get_latents_mask(mask, bs, height, width)
            if isinstance(inpaint_latents, torch.Tensor):
                inpaint_noise = torch.randn_like(inpaint_latents)
            else:
                latents = latents * mask + inpaint_latents[0].to(latents.device, latents.dtype) * (1 - mask)

        for i, (t_prev, t_curr) in enumerate(zip(timesteps[start:-1], timesteps[start+1:])):
            
            v_pred = self.get_v_prediction(
                latents,
                t_prev,
                cfg_strength,
                height,
                width,
                **kwargs
            )
            latents = latents + (t_curr - t_prev) * v_pred
            # If a single tensor is provided, we scale it to the same noise level by simply adding noise
            if mask is not None and isinstance(inpaint_latents, torch.Tensor):
                noisy_inpaint = t_prev * inpaint_noise + (1 - t_prev) * inpaint_latents
                latents = latents * mask + noisy_inpaint * (1 - mask)
            
            # If a list of latents is provided, they already correspond to different noise levels (possibly from inversion)
            elif mask is not None and isinstance(inpaint_latents, list):
                curr_inpaint = inpaint_latents[i+1]
                latents = latents * mask + curr_inpaint.to(latents.device, latents.dtype) * (1 - mask)
        
        return self.decode(latents, height, width, output_type='pil')

    @staticmethod
    def _pack_latents(latents: torch.Tensor, batch_size: int, num_channels_latents: int, height: int, width: int):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents: torch.Tensor, height: int, width: int, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

        return latents