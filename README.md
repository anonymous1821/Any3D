# Any3D
This repo contains the core implementation of generating [Any3D Dataset](https://huggingface.co/datasets/anonymous1821/Any3D), introduced in the anonymous submission "Any3D: Any3D: Push your 3D Diffusion Model towards Image Distribution" 

## Installation 
1. Clone the repo recursively (critical to get the submodule `PoissonRecon` cloned)
   ```bash
   git clone --recursive https://github.com/anonymous1821/Any3D.git
   ```
2. Install [Pytorch](https://pytorch.org/get-started/locally/) in conda env, we tested with `torch2.7.0+cu128`
   ```bash
   conda create -n any3d python=3.10 
   conda activate any3d 
   pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
   ```
3. Install other dependencies
   ```bash
   pip install diffusers==0.37.1 transformers==4.57.6 omegaconf einops plyfile objloader easydict matplotlib opencv-python pyvista==0.44.2 pysteps==1.12.0 open3d==0.19.0 cupy==13.5.1 xatlas==0.0.11 trimesh torch_scatter imageio[ffmpeg]  PyGLM==2.7.3 accelerate kornia timm

   pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/

   # For 2DGS Rasterization 
   pip install git+https://github.com/hbb1/diff-surfel-rasterization --no-build-isolation
   pip install git+https://github.com/camenduru/simple-knn --no-build-isolation
   pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
   pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
   ```

4. Build `PoissonRecon` (If you want to try Cycle View Model Enhancement)
   ```bash
   cd submodules/PoissonRecon
   make
   ```

5. Install `Blender`, we test with blender version 4.3.2 (If you want to preprocess your own data to try Cycle View Model Enhancement)
   ```bash
   wget https://download.blender.org/release/Blender4.3/blender-4.3.2-linux-x64.tar.xz
   tar -xvf blender-4.3.2-linux-x64.tar.xz
   rm -rf blender-4.3.2-linux-x64.tar.xz
   ```

6. We employ [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) for background removal, please apply for access to the gated huggingface repo. 


## Structure 
The structure of the repo is as following 
```bash
Any3D
  |-- configs # Configure the setting of system 
  |-- guidance # Implementation of the SDS-like score guidance, only RFDS is supported here as a minimal extraction 
  |-- inversion # Implementation of different inversion strategy for image editing
  |-- model # Implementation of different Text-to-Image Generation Model 
  |-- representations # Implementation of representations, only 2DGS is supported here 
  |-- sdedit # Implementation of a few SDEdit methods for image editing 
  |-- system # A system instantiate model, inversion or other algorithm from configs, handle the input, output. 
```

To run with most supported features, the working pipeline is load config --> instantiate system --> call system with inputs 
```python
from omegaconf import OmegaConf 
from utils.factory import instantiate_system
config = OmegaConf.load(...) # Config path
system = instantiate_system(config.System)
system(
    .... # Your Inputs
)
```

## Run
Currently this mainly support the Score Distillation Sampling (2DGS+RFDS+QwenImage) as described in the paper (use `configs/sds-2dgs-qwen`), and the Cycle-view enhancement (use `configs/cycle-enhance.yaml`)

> Note that these two methods both required a pre-generated model from [TRELLIS](https://github.com/microsoft/TRELLIS), you can refer to the official TRELLIS implementation to generate an inital 3D model (This may require more dependencies to build, please follow the instruction of TRELLIS). Also, since we use view-dependent prompt, which benefits from our canonical normalization, run `example_canonicalize.sh` first to orient the asset and write `front.png`, then `example_mv_prompts.py` for multi-view prompts.

### Canonical View Normalization
```bash
example_canonicalize.sh
```

### Score Distillation Sampling 
Input: A dictionary of Multi-view Prompts, a `ply` 3dgs checkpoint from TRELLIS, and the output root.
```bash
example_sds.sh
```
Run experiment with provided example 

### Cycle-View Enhancement 
Input: A directory that contains `textures`, `prompts_mv.json`, `mesh_processed.obj`, `mesh_processed.mtl`
If you want to preprocess your own data, refer to `preprocess.sh`
Run
```bash
example_cycle_enhance.sh
```
for experiment on provided example 
## Acknowledgement
This work builds upon a line of fantastic open sourced research works from the community. We sincerely thank:
[QwenImage](https://github.com/QwenLM/Qwen-Image), [FLUX.1](https://github.com/black-forest-labs/flux), [FLUX.2](https://github.com/black-forest-labs/flux2) for powerful opensourced image generation/editing models. 

[RF-Inversion](https://github.com/LituRout/RF-Inversion), [RF-Solver](https://github.com/wangjiangshan0725/RF-Solver-Edit), [FireFlow](https://github.com/HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing), [DNAEdit](https://github.com/xiechenxi99/DNAEdit_code), [UniEdit](https://github.com/DSL-Lab/UniEdit-Flow) for accurate inversion algorithm. 

[FlowEdit](https://github.com/fallenshock/FlowEdit) for inversion-free, heuristic image editing algorithm 

[RePaint](https://github.com/andreas128/RePaint) for image inpainting algorithm 

[ThreeStudio](https://github.com/DSL-Lab/UniEdit-Flow), [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian), [RFDS](https://github.com/yangxiaofeng/rectified_flow_prior) for implementation of SDS optimization. 

[2DGS](https://github.com/hbb1/2d-gaussian-splatting) for fantastic representation

[Elevate3D](https://github.com/ryunuri/Elevate3D), [PoissonRecon](https://github.com/mkazhdan/PoissonRecon), [continuous-remeshing](https://github.com/Profactor/continuous-remeshing) for mesh updating algorithm. 


