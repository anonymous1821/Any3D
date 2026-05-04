"""
Implementation of FLUX.1 Model (Specifically Dev) (https://github.com/black-forest-labs/flux)

Note that this implementation built upon diffusers version, which is known to have
a bit lower performance than the official implementation.
"""
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor 
from utils.registry import MODEL_REGISTRY
from .RFModel import RFModel 
from dataclasses import dataclass
from utils.config import BaseConfig
import numpy as np 
import inspect
import torch 
from typing import * 
@dataclass 
class FLUXConfig(BaseConfig):
    model_name: str 
    device: str = 'cuda:0'

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


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
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

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
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
    
@MODEL_REGISTRY.register('flux')
class FLUXModel(RFModel):
    Config = FLUXConfig
    
    def __init__(self, cfg: FLUXConfig):
        self.cfg = cfg
        model_name = cfg.model_name 
        self.device = cfg.device
        self.dtype = torch.bfloat16
        pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=self.dtype)
        pipe.to(self.device)

        self.vae = pipe.vae 
        self.text_encoder = pipe.text_encoder 
        self.tokenizer = pipe.tokenizer 
        self.text_encoder_2 = pipe.text_encoder_2
        self.tokenizer_2 = pipe.tokenizer_2
        self.transformer = pipe.transformer
        self.scheduler = pipe.scheduler 

        del pipe 

        self.latents_mean = self.vae.config.shift_factor 
        self.latents_std = self.vae.config.scaling_factor
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.max_sequence_length = 512
        self.num_channels_latents = self.transformer.config.in_channels // 4 
    
    def to(self, device):
        self.device = device
        self.transformer = self.transformer.to(device)
        self.text_encoder = self.text_encoder.to(device)
        self.text_encoder_2 = self.text_encoder_2.to(device)
        self.vae = self.vae.to(device)
        return self
    
    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    @staticmethod
    def _get_latent_image_ids(bs, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None
    ):
     

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            log.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {self.max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(self.device), output_hidden_states=False)[0]

        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)

        _, seq_len, _ = prompt_embeds.shape

        prompt_embeds = prompt_embeds.view(batch_size, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]]
    ):
        device = self.text_encoder.device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            log.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)

        prompt_embeds = prompt_embeds.view(batch_size, -1)

        return prompt_embeds


    def encode(
        self, 
        x: torch.Tensor,
        generator: Optional[torch.Generator] = None
    ):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.to(self.vae.device, self.vae.dtype)
        bs = x.shape[0]
        x = x * 2. - 1.
        latents = retrieve_latents(self.vae.encode(x), generator = generator)
        latents = (latents - self.latents_mean) * self.latents_std

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
        latents = (latents / self.latents_std) + self.latents_mean
        x = self.vae.decode(latents, return_dict=False)[0]
        x = self.image_processor.postprocess(x, output_type = output_type)
        return x 
    
    def set_prompt(
        self,
        prompt, 
        negative_prompt = ""
    ):
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * len(prompt)
        bs = len(prompt)
        prompts = prompt + negative_prompt 
        pooled_prompt_embeds = self._get_clip_prompt_embeds(prompts)
        prompt_embeds = self._get_t5_prompt_embeds(prompts)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=self.device, dtype=self.dtype)
        self.cond = {
            "pooled_projections": pooled_prompt_embeds, 
            "encoder_hidden_states": prompt_embeds, 
            "txt_ids": text_ids
        }
        pos_pooled_prompt_embeds = pooled_prompt_embeds.chunk(2)[0]
        pos_prompt_embeds = prompt_embeds.chunk(2)[0]
        pos_text_ids = torch.zeros(pos_prompt_embeds.shape[1], 3).to(device=self.device, dtype=self.dtype)
        self.pos_cond = {
            "pooled_projections": pos_pooled_prompt_embeds, 
            "encoder_hidden_states": pos_prompt_embeds, 
            "txt_ids": pos_text_ids
        }

        return self.cond, self.pos_cond

    def get_v_prediction(
        self,
        latents: torch.Tensor, 
        t: float, 
        cfg_strength: float = 1.0, 
        guidance_scale: float =3.5, 
        width: int = None, 
        height: int = None,
        latent_image_ids = None,
        cond: Dict = None,
        pos_cond: Dict = None, 
        **kwargs
    ):
        do_cfg = cfg_strength > 1. 
        bs = latents.shape[0]
        t = torch.full((bs,), t, device=self.device, dtype=self.dtype)
        if self.transformer.config.guidance_embeds:
            guidance = torch.full((bs, ), guidance_scale, device=self.device, dtype=self.dtype)
        else:
            guidance = None
        if not latent_image_ids:
            assert (width and height)
            height = 2 * (int(height) // (self.vae_scale_factor * 2))
            width = 2 * (int(width) // (self.vae_scale_factor * 2))
            latent_image_ids = self._get_latent_image_ids(bs, height // 2, width // 2, self.device, torch.bfloat16)
        
        if do_cfg:
            cond = self.cond if cond is None else cond
            v_pred, neg_v_pred = self.transformer(
                hidden_states = latents, 
                timestep = t, 
                **cond, 
                guidance=guidance,
                img_ids=latent_image_ids, 
                return_dict = False,
            )[0].chunk(2)
            v_pred = neg_v_pred + cfg_strength * (v_pred - neg_v_pred)
        else:
            cond = self.pos_cond if pos_cond is None else pos_cond
            v_pred = self.transformer(
                hidden_states = latents, 
                timestep = t, 
                **cond, 
                guidance=guidance,
                img_ids=latent_image_ids, 
                return_dict = False,
            )[0]

        return v_pred 
    
    def get_schedule(
        self, 
        num_steps: int, 
        width: int, 
        height: int,
        **kwargs
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        image_seq_len = (height // 2) * (width // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_steps,
            self.device,
            sigmas = sigmas, 
            mu=mu,
        )

        timesteps /= 1000 
        timesteps = torch.cat([timesteps, torch.tensor([0.], device=timesteps.device)])
        sigmas = self.scheduler.sigmas.to(self.device).float()
        return timesteps, sigmas

    @torch.no_grad()
    def sample(
        self, 
        prompt: str | List[str],
        negative_prompt: str = "",
        num_steps: int = 50, 
        cfg_strength: float = 1.0,
        guidance_scale: float = 3.5,
        height: int = 1024, 
        width: int = 1024,
        generator: torch.Generator = None,
        start: int = 0, 
        latents: torch.Tensor = None, 
        mask: torch.Tensor = None, # 1 for editied region, 0 for static region
        inpaint_latents: torch.Tensor | List[torch.Tensor] = None, 
        **kwargs
    ):
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
            latents = self._pack_latents(latents, num_images, latent_height, latent_width)
        timesteps, _ = self.get_schedule(num_steps, width, height)
        
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[start:-1], timesteps[start+1:])):
            # Inpainting 
            # If a single tensor is provided, we scale it to the same noise level by simply adding noise
            if mask is not None and isinstance(inpaint_latents, torch.Tensor):
                noisy_inpaint_latents = t_curr * torch.randn_like(inpaint_latents) + (1 - t_prev) * inpaint_latents
                latents = latents * mask + noisy_inpaint_latents * (1 - mask)

            # If a list of latents is provided, they already correspond to different noise levels (possibly from inversion)
            if mask is not None and isinstance(inpaint_latents, list):
                latents = latents * mask + inpaint_latents[i] * (1 - mask)
            
            v_pred = self.get_v_prediction(
                latents,
                t_prev,
                cfg_strength = cfg_strength,
                guidance_scale = guidance_scale,
                height = height,
                width = width,
                **kwargs
            )
            latents = latents + (t_prev - t_curr) * v_pred
        return self.decode(latents, height, width, output_type='pil')