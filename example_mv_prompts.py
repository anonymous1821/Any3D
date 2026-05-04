from utils.prompts.sys_prompts import MVPrompts4Views, MVPrompts6Views
from utils.assistants.initialize_models import init_model
import argparse 
import re
import json 
def extract_json(text):
    pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    pattern = r"(\{[\s\S]*\})"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    
    return text

def main(args):
    device = 'cuda:0'
    model, processor = init_model(args.model)
    model = model.to(device)
    with open(args.input_path, 'r') as f:
        prompt = f.read().strip()
    if args.num_views == 4:
        SystemPrompt = MVPrompts4Views
    else:
        SystemPrompt = MVPrompts6Views
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SystemPrompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"The text prompt is: {prompt}"},
                {"type": "text", "text": "Front reference image:"},
                {"type": "image", "image": args.front_path},
                {"type": "text", "text": "Generate multi-view prompts following the specified output JSON format."}
            ]
        }
    ]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inputs = inputs.to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    output_text = extract_json(output_text[0])
    output = json.loads(output_text)
    with open(args.output_path, 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--front_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-VL-32B-Instruct')
    parser.add_argument('--num_views', type=int, default=4, choices=[4, 6])
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    args = parser.parse_args() 
    main(args)