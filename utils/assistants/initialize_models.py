"""
Initialize Large Vision Language Model or Large Language Model
Currently support Qwen3.5, Qwen3, Qwen3-VL, Intern-VL series 
"""
import torch
from transformers import (
    AutoProcessor, 
    AutoTokenizer,
    Qwen3VLForConditionalGeneration, 
    AutoModelForCausalLM,
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForImageTextToText
)
try:
    from transformers import Qwen3_5ForConditionalGeneration
except:
    pass 
try:
    from transformers import Glm4vForConditionalGeneration
except:
    pass 


    

def init_model(model_name: str, device='cuda'):
    if "Qwen2.5-VL" in model_name:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map=None,
            torch_dtype="auto",
            attn_implementation="flash_attention_2"
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor
    elif "Qwen3.5" in model_name:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_name,
            device_map=None,
            torch_dtype="auto",
            # attn_implementation="flash_attention_2"
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor
    elif "Qwen3-VL" in model_name:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map=None,
            torch_dtype="auto",
            attn_implementation="flash_attention_2"
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor
    elif "GLM" in model_name:
        model = Glm4vForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None,
            torch_dtype="auto",
            attn_implementation="flash_attention_2"
        )
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer 
    