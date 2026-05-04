"""
Implementation of Qwen-Image-Edit model (https://github.com/QwenLM/Qwen-Image)
Please cite the original paper "Qwen-Image Technical Report" (arXiv 2025)
if you find this model helpful

Note that this "plus" model is for 2509 and 2511 version of Qwen-Image-Edit
"""
from operator import iadd
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
from PIL import Image
import math  
from tqdm import tqdm
CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024

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

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height

@dataclass
class QwenImageModelConfig(BaseConfig):
    model_name: str
    device: str='cuda:0'

@MODEL_REGISTRY.register('qwen-image-edit-plus')
class QwenImageEditPlusModel(RFModel):
    Config = QwenImageModelConfig
    def __init__(self, cfg: QwenImageModelConfig):
        self.cfg = cfg 
        model_name = cfg.model_name 
        device = self.device = cfg.device
        self.dtype = torch.bfloat16 

        pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=self.dtype)
        pipe.to(device)

        self.vae = pipe.vae 
        self.text_encoder = pipe.text_encoder 
        self.tokenizer = pipe.tokenizer 
        self.processor = pipe.processor
        self.transformer = pipe.transformer 
        self.scheduler = pipe.scheduler 
        del pipe 

        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.latent_channels = self.vae.config.z_dim if getattr(self, "vae", None) else 16

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = 1024

        self.prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        self.prompt_template_encode_start_idx = 64
        self.default_sample_size = 128
        self.max_sequence_length = 1024
        self.latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(self.device, self.dtype)
        )
        self.latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(self.device, self.dtype)
        )
        self.num_channels_latents = self.transformer.config.in_channels // 4 

    def to(self, device):
        self.vae = self.vae.to(device)
        self.text_encoder = self.text_encoder.to(device)
        self.transformer = self.transformer.to(device)
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
        prompt: str | List[str],
        image: torch.Tensor | List[torch.Tensor] = None
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if isinstance(image, list):
            base_img_prompt = ""
            for i, img in enumerate(image):
                base_img_prompt += self.img_prompt_template.format(i + 1)
        elif image is not None:
            base_img_prompt = self.img_prompt_template.format(1)
        else:
            base_img_prompt = ""


        template = self.prompt_template_encode

        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(base_img_prompt + e) for e in prompt]


        
        model_inputs = self.processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)

        return prompt_embeds, encoder_attention_mask

    def get_latents_mask(
        self,
        mask: torch.Tensor | List[torch.Tensor], 
        batch_size: int,
        height: int,
        width: int,
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
        latents = retrieve_latents(self.vae.encode(x), generator = generator, sample_mode="argmax")
        
        latents = (latents - self.latents_mean) / self.latents_std

        latent_height, latent_width = latents.shape[-2:]
        num_channels_latents = self.transformer.config.in_channels // 4 
        latents = self._pack_latents(latents, bs, num_channels_latents, latent_height, latent_width)

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
        x = self.vae.decode(latents.to(self.dtype), return_dict=False)[0][:, :, 0]
        x = self.image_processor.postprocess(x, output_type = output_type)
        return x 
    
    def set_prompt(
        self, 
        prompt: str | List[str],
        negative_prompt = "",
        image: torch.Tensor | List[torch.Tensor] = None
    ):
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        prompts = prompt + negative_prompt
        bs = len(prompt)
        
        prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, image)
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.view(bs, seq_len)

        self.pos_cond = {
            'encoder_hidden_states': prompt_embeds, 
            'encoder_hidden_states_mask': prompt_embeds_mask,
        }

        prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(negative_prompt, image)
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.view(bs, seq_len)

        self.neg_cond = {
            'encoder_hidden_states': prompt_embeds, 
            'encoder_hidden_states_mask': prompt_embeds_mask,
        }

        return self.pos_cond, self.neg_cond
    
    def prepare_image_latents(
        self,
        images,
        height,
        width,
        generator,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        if not isinstance(images, list):
            images = [images]
        all_image_latents = []
        for image in images:
            image = image.to(device=self.device, dtype=self.dtype)
            image_latents = self.encode(x=image, generator=generator)
            image_latents = torch.cat([image_latents], dim=0)

            all_image_latents.append(image_latents)
        image_latents = torch.cat(all_image_latents, dim=1)

        return image_latents
    
    def preprocess_image(self, image):
        if not isinstance(image, list):
            image=[image]
        if len(image[0].shape) == 3:
            image = [img.unsqueeze(0) for img in image]
        condition_image_sizes = []
        condition_images = []
        vae_image_sizes = []
        vae_images = []
        for img in image:
            image_width, image_height = img.shape[-2:]
            condition_width, condition_height = calculate_dimensions(
                CONDITION_IMAGE_SIZE, image_width / image_height
            )
            vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
            condition_image_sizes.append((condition_width, condition_height))
            vae_image_sizes.append((vae_width, vae_height))
            condition_images.append(self.image_processor.resize(img, condition_height, condition_width))
            vae_images.append(self.image_processor.resize(img, vae_height, vae_width).unsqueeze(2))
        return condition_images, vae_images, vae_image_sizes

    def get_v_prediction(
        self,
        latents: torch.Tensor,
        image_latents: torch.Tensor,
        t: torch.Tensor, 
        cfg_strength: float,
        width: int, 
        height: int, 
        vae_image_sizes: List[Tuple[int, int]],
        img_shapes: List[List[Tuple[int, int, int]]] = None,
        pos_cond: Dict = None, 
        neg_cond: Dict = None,
        **kwargs
    ):
        bs = latents.shape[0]
        if img_shapes is None:
            img_shapes = [
                [
                    (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                    *[
                        (1, vae_height // self.vae_scale_factor // 2, vae_width // self.vae_scale_factor // 2)
                        for vae_width, vae_height in vae_image_sizes
                    ],
                ]
            ] * bs
            
        do_cfg = cfg_strength > 1. 
        t = torch.tensor(t).repeat(bs).to(self.device, self.dtype)
        input_latents = torch.cat([latents, image_latents], dim = 1)
        if do_cfg:
            cond = self.pos_cond if pos_cond is None else pos_cond
            v_pred=self.transformer(
                hidden_states=input_latents.to(self.dtype),
                timestep=t,
                **cond,
                img_shapes=img_shapes * bs,
                return_dict=False,
            )[0]
            cond = self.neg_cond if neg_cond is None else neg_cond
            neg_v_pred=self.transformer(
                hidden_states=input_latents.to(self.dtype),
                timestep=t,
                **cond,
                img_shapes=img_shapes * bs,
                return_dict=False,
            )[0]

            v_pred = v_pred[:, : latents.size(1)]
            neg_v_pred = neg_v_pred[:, : latents.size(1)]
            comb_pred = neg_v_pred + cfg_strength * (v_pred - neg_v_pred)
            cond_norm = torch.norm(v_pred, dim=-1, keepdim=True)
            comb_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            v_pred = comb_pred * (cond_norm / comb_norm)
        
        else:
            cond = self.pos_cond if pos_cond is None else pos_cond
            v_pred = self.transformer(
                hidden_states = input_latents.to(self.dtype),
                timestep = t,
                **cond,
                img_shapes = img_shapes * bs,
                return_dict = False
            )[0]
            v_pred = v_pred[:, : latents.size(1)]
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
        
        sigmas = self.scheduler.sigmas.to(self.device).float()

        return timesteps, sigmas

    @torch.no_grad()
    def sample(
        self, 
        prompt: str | List[str],
        image: torch.Tensor | List[torch.Tensor],
        negative_prompt: str | List[str] = "",
        num_steps: int = 50,
        cfg_strength: float = 4.0,
        height: int = 1024,
        width: int = 1024,
        generator = None,
        start: int = 0, 
        latents: torch.Tensor | List[torch.Tensor]= None, 
        mask: torch.Tensor | List[torch.Tensor]= None, # 1 for editied region, 0 for static region
        inpaint_latents: torch.Tensor | List[torch.Tensor] = None, 
        verbose: bool = True,
        **kwargs
    ):
        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of
        bs = len(prompt) if isinstance(prompt, list) else 1
        latent_height = 2 * (int(height) // (self.vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (self.vae_scale_factor * 2))
        condition_images, vae_images, vae_image_sizes = self.preprocess_image(image)
    
        self.set_prompt(prompt, negative_prompt, condition_images)
        image_latents = self.prepare_image_latents(
            vae_images,
            height,
            width,
            generator,
        )
        if latents is None:
            latents = torch.randn(
                bs, 
                self.num_channels_latents, 
                latent_height, 
                latent_width,
                device=self.device, 
                dtype=self.dtype,
                generator=generator
            ) 
        if len(latents.shape) > 3:
            latents = self._pack_latents(latents, bs, self.num_channels_latents, latent_height, latent_width)

        timesteps, _ = self.get_schedule(num_steps, width, height)
        timesteps = torch.cat([timesteps, torch.tensor([0.], device=timesteps.device)])
        # Prepare inpainting noise
        inpaint_noise = None
        if mask is not None:
            mask = [mask] if not isinstance(mask, list) else mask
            mask = [m.to(self.device, self.dtype) for m in mask]
            mask = self.get_latents_mask(mask, bs, height, width)
            if isinstance(inpaint_latents, torch.Tensor):
                inpaint_noise = torch.randn_like(inpaint_latents)
            else:
                latents = latents * mask + inpaint_latents[0].to(latents.device, latents.dtype) * (1 - mask)

        for i, (t_prev, t_curr) in tqdm(enumerate(zip(timesteps[start:-1], timesteps[start+1:])), desc="Sampling by QwenImageEditPlusModel", disable = not verbose):
            
            v_pred = self.get_v_prediction(
                latents=latents,
                image_latents=image_latents,
                t=t_prev, 
                cfg_strength=cfg_strength,
                width=width, 
                height=height, 
                vae_image_sizes=vae_image_sizes
            )
            latents = latents + (t_curr - t_prev) * v_pred
            
            # If a single tensor is provided, we scale it to the same noise level by simply adding noise
            if mask is not None and isinstance(inpaint_latents, torch.Tensor):
                noisy_inpaint = t_prev * inpaint_noise + (1 - t_prev) * inpaint_latents
                latents = latents * mask + noisy_inpaint * (1 - mask)
            
            # If a list of latents is provided, they already correspond to different noise levels (possibly from inversion)
            elif mask is not None and isinstance(inpaint_latents, list):
                curr_inpaint = inpaint_latents[i+1]
                if len(curr_inpaint.shape) > 3:
                     curr_inpaint, _ = batched_prc_img(curr_inpaint)
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