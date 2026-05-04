import torch
import torch.nn as nn
from einops import rearrange
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Mistral3ForConditionalGeneration,
)

from .system_messages import SYSTEM_MESSAGE

OUTPUT_LAYERS_MISTRAL = [10, 20, 30]
OUTPUT_LAYERS_QWEN3 = [9, 18, 27]
MAX_LENGTH = 512


class Mistral3SmallEmbedder(nn.Module):
    def __init__(
        self,
        model_spec: str = "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        model_spec_processor: str = "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        torch_dtype: str = "bfloat16",
    ):
        super().__init__()

        self.model: Mistral3ForConditionalGeneration = Mistral3ForConditionalGeneration.from_pretrained(
            model_spec,
            torch_dtype=getattr(torch, torch_dtype),
        )
        self.processor = AutoProcessor.from_pretrained(model_spec_processor, use_fast=False)

        self.max_length = MAX_LENGTH

    def format_input(
        self,
        txt: list[str],
        system_message: str = SYSTEM_MESSAGE,
    ) -> list[list[dict]]:
        """
        Format a batch of text prompts into the conversation format expected by apply_chat_template.

        Args:
            txt: List of text prompts
            system_message: System message to use (default: CREATIVE_SYSTEM_MESSAGE)

        Returns:
            List of conversations, where each conversation is a list of message dicts
        """
        # Remove [IMG] tokens from prompts to avoid Pixtral validation issues
        # when truncation is enabled. The processor counts [IMG] tokens and fails
        # if the count changes after truncation.
        cleaned_txt = [prompt.replace("[IMG]", "") for prompt in txt]

        return [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            for prompt in cleaned_txt
        ]


    @torch.no_grad()
    def forward(self, txt: list[str]):
        # Format input messages
        messages_batch = self.format_input(txt=txt)

        # Process all messages at once
        # with image processing a too short max length can throw an error in here.
        inputs = self.processor.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        # Move to device
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        # Forward pass through the model
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS_MISTRAL], dim=1)
        return rearrange(out, "b c l d -> b l (c d)")


class Qwen3Embedder(nn.Module):
    def __init__(
        self,
        model_spec: str,
        device: str | torch.device = "cuda",
    ):
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_spec,
            torch_dtype=None,
            device_map=str(device),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_spec)
        self.max_length = MAX_LENGTH

    @torch.no_grad()
    def forward(self, txt: list[str]):
        all_input_ids = []
        all_attention_masks = []

        for prompt in txt:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            model_inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

            all_input_ids.append(model_inputs["input_ids"])
            all_attention_masks.append(model_inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(self.model.device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(self.model.device)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3], dim=1)
        return rearrange(out, "b c l d -> b l (c d)")


def load_mistral_small_embedder(device: str | torch.device = "cuda") -> Mistral3SmallEmbedder:
    return Mistral3SmallEmbedder().to(device)


def load_qwen3_embedder(variant: str, device: str | torch.device = "cuda"):
    return Qwen3Embedder(model_spec=f"Qwen/Qwen3-{variant}-FP8", device=device)
