from omegaconf import OmegaConf 
from utils.factory import instantiate_system
import argparse
def main(args):
    # Pay special attention, we currently support device like 'cuda:0' but not 'cuda'
    device='cuda:0'
    config = OmegaConf.load('./configs/cycle-enhance.yaml') # Config path
    system = instantiate_system(config.System)
    system = system.to(device)

    # data_path 
    #    |-- textures
    #    |-- prompts_mv.json 
    #    |-- mesh_processed.obj 
    #    |-- mesh_processed.mtl 
    system(
        data_path = args.data_path, 
        verbose = True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./examples/cycle_enhance/example', help = 'Path to save the output')
    args = parser.parse_args()
    main(args)