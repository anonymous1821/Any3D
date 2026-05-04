from omegaconf import OmegaConf 
from utils.factory import instantiate_system
import json 
import argparse
def main(args):
    # Pay special attention, we currently support device like 'cuda:0' but not 'cuda'
    device='cuda:0'
    config = OmegaConf.load('./configs/sds-2dgs-qwen.yaml') # Config path
    system = instantiate_system(config.System)
    system = system.to(device)
    with open(args.prompts_path, 'r') as f:
        prompts = json.load(f)
    system(
        prompts=prompts, 
        checkpoint=args.checkpoint_path, 
        output_path=args.output_path, 
        verbose=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required = True, help = 'Path to 3dgs checkpoint')
    parser.add_argument('--prompts_path', required = True, help = 'Path to prompts json file')
    parser.add_argument('--output_path', required = True, help = 'Path to save the output')
    args = parser.parse_args()
    main(args)