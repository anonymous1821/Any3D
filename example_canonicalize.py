from utils.prompts.sys_prompts import FrontViewSelect
from utils.assistants.initialize_models import init_model
from utils.canonical import ORTHO_YAWS, ORTHO_PITCHES, apply_canonical_rotation, ortho_index_to_yaw_pitch
from utils.factory import instantiate_renderer
from omegaconf import OmegaConf
from PIL import Image
import argparse
import glob
import os
import re
import shutil
import torch


def main(args):
    # Pay special attention, we currently support device like 'cuda:0' but not 'cuda'
    device = 'cuda:0'
    with open(args.prompt_path, 'r') as f:
        prompt = f.read().strip()

    ortho_dir = os.path.join(os.path.dirname(args.output_path) or '.', 'ortho_render')
    view_paths = sorted(glob.glob(os.path.join(ortho_dir, '*.png')))
    if len(view_paths) == 0:
        os.makedirs(ortho_dir, exist_ok=True)
        cfg = OmegaConf.create({
            'type': 'Gaussian2d',
            'num_pts': None,
            'bg_color': [0.5, 0.5, 0.5],
            'gaussian_config': {'aabb': [-0.5, -0.5, -0.5, 1.0, 1.0, 1.0], 'sh_degree': 0},
        })
        renderer = instantiate_renderer(cfg, device=device)
        renderer.initialize(args.checkpoint_path)
        renderer.representation.active_sh_degree = renderer.representation.max_sh_degree
        with torch.no_grad():
            for i, (yaw, pitch) in enumerate(zip(ORTHO_YAWS, ORTHO_PITCHES)):
                out = renderer.render(yaw, pitch, 2.0, 40.0, width=512, height=512)
                rgb = out['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                path = os.path.join(ortho_dir, f'{i:02d}_yaw{yaw}_pitch{pitch}.png')
                Image.fromarray((rgb * 255).astype('uint8')).save(path)
                view_paths.append(path)
    else:
        view_paths = view_paths[:8]

    model, processor = init_model(args.model)
    model = model.to(device)
    messages = [
        {
            'role': 'system',
            'content': [{'type': 'text', 'text': FrontViewSelect}],
        },
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': f'The text prompt is: {prompt}'},
                {'type': 'text', 'text': 'Orthogonal Rendering from the 3D model:'},
            ]
            + [{'type': 'image', 'image': vp} for vp in view_paths]
            + [{'type': 'text', 'text': "Select the single image that shows the 'front' view. Reply with ONLY one integer 1-8 corresponding to the order above."}],
        },
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt'
    )
    inputs = inputs.to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    idx_0 = int(re.search(r'\b([1-8])\b', output_text).group(1)) - 1
    sel_yaw, sel_pitch = ortho_index_to_yaw_pitch(idx_0)

    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    shutil.copy(view_paths[idx_0], args.output_path)
    apply_canonical_rotation(
        sel_yaw=sel_yaw,
        sel_pitch=sel_pitch,
        ply_paths=[args.checkpoint_path],
        glb_paths=[args.glb_path] if args.glb_path else [],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True, help='Path to 3dgs checkpoint')
    parser.add_argument('--prompt_path', required=True, help='Path to prompt text file')
    parser.add_argument('--output_path', required=True, help='Path to save front.png')
    parser.add_argument('--glb_path', type=str, default=None, help='Optional GLB to rotate')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-VL-32B-Instruct')
    parser.add_argument('--max_new_tokens', type=int, default=128)
    args = parser.parse_args()
    main(args)
