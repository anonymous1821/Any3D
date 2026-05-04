"""
This is largely built on Elevate3D (https://github.com/ryunuri/Elevate3D)
Please cite the original paper "Elevating 3D Models: High-Quality Texture and Geometry Refinement from a Low-Quality Model" (SIGGRAPH 2025)
if you find this algorithm helpful 
"""
from pathlib import Path
from utils.registry import SYSTEM_REGISTRY 
from utils.factory import instantiate_system
from dataclasses import dataclass 
from typing import * 
from omegaconf import OmegaConf
from objloader import Obj
from model.marigold.pipe import MarigoldNormalsPipeline
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from utils.Poisson import(
    convert_obj_to_ply,
    convert_ply_to_obj,
    process_poisson,
    render_depth_prior, 
    run_bilateral_integration
)
from utils.Poisson.normal_blending import load_and_blend_maps
from utils.Poisson.mesh import Mesh
from utils.smooth import L0Smoothing
from utils.Projection import HeadlessProjectionMapping, HeadlessBaker
from enum import Enum 
from PIL import Image 
from utils.config import BaseConfig 
from utils.common import seed_everything
import numpy as np 
import shutil
import torch 
import time
import pymeshlab
import json
from functools import partial
from utils.etc import (
    normalize_angle,
    move_existing_files,
    dilate_mask,
    copy_texture_files,
)
from utils.logger import setup_logging, get_logger
setup_logging()
log = get_logger(__name__)

class RefineType(Enum):
    SIGNIFICANT = 0
    NEGLIGIBLE = 1
    SKIP = 2

@dataclass
class CameraAngle:
    yaw: float
    pitch: float

@dataclass
class CycleEnhance3DConfig(BaseConfig):
    ImageSystem: BaseConfig
    normal: BaseConfig
    projection: BaseConfig
    poisson: BaseConfig
    camera_schedule: List[Dict[str, float]]

    negative_prompt: str = ""        
    im_res: int = 1024
    cos_thresh: float = 0.5
    negligible_thresh: float = 0.02
    significant_thresh: float = 0.05
    bake: bool = True
    type: str = 'cycle-enhance-3d'
    seed: Optional[int] = None
    prompt_template: str = ""

    # Whether to use the view dependent prompts
    enable_mv_prompts: bool = False

@SYSTEM_REGISTRY.register('cycle-enhance-3d')
class CycleEnhance3DSystem:
    Config = CycleEnhance3DConfig
    def __init__(self, cfg: CycleEnhance3DConfig):
        self.cfg = cfg
        if self.cfg.seed is not None:
            seed_everything(self.cfg.seed)

        # Ensure nested configs are accessible via dot notation
        for key in ['normal', 'projection', 'poisson']:
            if hasattr(self.cfg, key) and isinstance(getattr(self.cfg, key), dict):
                setattr(self.cfg, key, OmegaConf.create(getattr(self.cfg, key)))

        self.image_refiner = instantiate_system(cfg.ImageSystem)
        # Normal Esitmation Model (Marigold)
        self.normal_pipe = MarigoldNormalsPipeline.from_pretrained(
            cfg.normal.model_name, variant=None, dtype=torch.float32
        )
        try:
            self.normal_pipe.enable_xformers_memory_efficient_attention()
        except ImportError:
            pass
        self.normal_estimator = partial(
            self.normal_pipe,
            denoising_steps=self.cfg.normal.num_steps,
            ensemble_size=self.cfg.normal.ensemble_size,
            processing_res=self.cfg.normal.processing_res,
            match_input_res=True,
            show_progress_bar=False,  # Disable internal progress bar
            resample_method=self.cfg.normal.resample_method
        )
        self.bg_remover = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-2.0",
            trust_remote_code=True
        ).eval()
        self.bg_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.camera_schedule = cfg.camera_schedule
        # Use list of objects for iteration
        self.camera_angles = [CameraAngle(yaw=entry['yaw'] % 360, pitch=entry['pitch']) for entry in self.camera_schedule]

        self.device = 'cpu'
        self.device_idx = None

        
    def to(self, device):
        self.image_refiner = self.image_refiner.to(device)
        self.normal_pipe = self.normal_pipe.to(device)
        self.bg_remover = self.bg_remover.to(device)
        self.device = device
        # Extract integer device ID if device is a string
        if isinstance(self.device, str) and ':' in self.device:
            self.device_idx = int(self.device.split(':')[-1])
        elif isinstance(self.device, int):
            self.device_idx = self.device
        else:
            self.device_idx = 0
        return self

    @torch.no_grad()
    def remove_bg(
        self,
        image
    ):
        init_size = image.size 
        input = self.bg_transform(image)
        input = input.to(self.device).unsqueeze(0)
        preds = self.bg_remover(input)[-1][0].sigmoid().cpu()
        pred_pil = transforms.ToPILImage()(preds)
        pred_pil = pred_pil.resize(init_size)
        image.putalpha(pred_pil)
        return image 
    

    @staticmethod
    def compute_refinement_ratio(renderer, pitch, yaw, res=1024, thresh=0.7):
        """
        Render just enough to compute a ratio of how much needs refinement
        (low cosine values) vs. the total foreground.
        """
        image = renderer.render(
            pitch=180 + pitch, yaw=yaw, img_res=(res, res), zoom=1.0
        )
        zoom = renderer.adjust_camera_zoom(np.array(image), res, desired_ratio=0.90)

        # Render necessary images for ratio calculation
        render_kwargs = {
            "pitch": 180 + pitch,
            "yaw": yaw,
            "img_res": (res, res),
            "zoom": zoom,
        }

        normal_image  = renderer.render_normal(**render_kwargs)
        cosine_image  = renderer.render_cosine(**render_kwargs)
        cam_cos_image = renderer.render_cam_cos(**render_kwargs)

        cosine_thresh_mask = renderer.calculate_low_cosine_similarity_mask(
            cosine_image, thresh
        )
        fg_pil = normal_image.split()[-1]
        cosine_thresh_mask = dilate_mask(
            np.array(cosine_thresh_mask), np.array(fg_pil)
        )

        # Compute the ratio of the inpaint mask to the foreground object
        fg_mask = np.array(fg_pil) > 0
        inpaint_mask = np.array(cosine_thresh_mask) > 0
        
        np_cam_cos_map = np.array(cam_cos_image) / 255.
        cos_weighted_inpaint_area = np_cam_cos_map[inpaint_mask].sum()
        fg_area = np.count_nonzero(fg_mask)
        
        if fg_area == 0:
            ratio = 0.0
        else:
            ratio = cos_weighted_inpaint_area / fg_area

        if ratio > self.cfg.negligible_thresh:
            refine_type = RefineType.SIGNIFICANT
        else:
            refine_type = RefineType.SKIP

        return ratio, refine_type

    @staticmethod
    def render_planar_projection(
        cfg,
        obj_mesh,
        texture_dir,
        pitch,
        yaw,
        res=1024,
        zoom=1.0,
        thresh=0.7,
        device_idx=None,
    ):
        """
        Static utility method that renders a mesh from a given pitch/yaw and
        returns the zoom factor, refinement type, etc.
        """
        renderer = HeadlessProjectionMapping(
            vertex_shader_path=cfg.projection.vertex_shader_path,
            normal_fragment_shader_path=cfg.projection.normal_fragment_shader_path,
            obj_mesh=obj_mesh,
            texture_dir=texture_dir,
            device_idx=device_idx,
        )

        # Initial render to calculate zoom
        image = renderer.render(pitch=180 + pitch, yaw=yaw, img_res=(res, res), zoom=1.0)
        
        zoom = renderer.adjust_camera_zoom(np.array(image), res, desired_ratio=0.90)

        # Define common rendering arguments
        render_kwargs = {
            "pitch": 180 + pitch,
            "yaw": yaw,
            "img_res": (res, res),
            "zoom": zoom,
        }

        # List of rendering functions and corresponding keys
        render_funcs = [
            ("render", "rgba"),
            ("render_normal", "normal"),
            ("render_depth", "depth"),
            ("render_cosine", "cosine"),
            ("render_cam_cos", "cam_cos"),
        ]

        # Dictionary to store rendered images
        rendered_images = {}

        # Render images
        for func_name, key in render_funcs:
            render_func = getattr(renderer, func_name)
            rendered_images[key] = render_func(**render_kwargs)

        # Get the cosine image for further processing
        cosine_image = rendered_images["cosine"]
        cosine_thresh_mask = renderer.calculate_low_cosine_similarity_mask(cosine_image, thresh)
        
        fg_pil = rendered_images["normal"].split()[-1]
        
        cosine_thresh_mask = dilate_mask(
            np.array(cosine_thresh_mask), np.array(fg_pil)
        )

        # Compute the ratio of the inpaint mask to the foreground object
        fg_mask = np.array(fg_pil) > 0
        inpaint_mask = np.array(cosine_thresh_mask) > 0

        np_cam_cos_map = np.array(rendered_images["cam_cos"]) / 255.
        cos_weighted_inpaint_area = np_cam_cos_map[inpaint_mask].sum()
        fg_area = np.count_nonzero(fg_mask)

        if fg_area == 0:
            ratio = 0.0
        else:
            ratio = cos_weighted_inpaint_area / fg_area

        if ratio >= cfg.significant_thresh:
            refine_type = RefineType.SIGNIFICANT
        elif ratio > cfg.negligible_thresh:
            refine_type = RefineType.NEGLIGIBLE
        else:
            refine_type = RefineType.SKIP

        del renderer
        
        return zoom, refine_type, ratio, cosine_thresh_mask, rendered_images

    def get_prompt(self, angle: CameraAngle):
        if self.cfg.enable_mv_prompts:
            yaw = normalize_angle(angle.yaw)
            pitch = angle.pitch
            
            # NOTE: We assume yaw = 180 corresponds to the front
            # if pitch >= 45:
            #     direction = 'overhead'
            # elif 135 <= yaw <= 225:
            #     direction = 'front'
            # elif 225 < yaw < 315:
            #     direction = 'left'
            # elif (315 <= yaw <= 360) or (0 <= yaw <= 45):
            #     direction = 'back'
            # else:
            #     direction = 'right'
            if pitch >= 45:
                direction = 'overhead'
            elif pitch < -30:
                direction = None
            elif 170 <= yaw <= 190:
                direction = 'front'
            elif 190 < yaw < 260:
                direction = 'front-left'
            elif 280 < yaw < 350:
                direction = 'back-left'
            elif (350 <= yaw <= 360) or (0 <= yaw <= 10):
                direction = 'back'
            elif 10 < yaw < 80:
                direction = 'back-right'
            elif 100 < yaw < 170:
                direction = 'front-right'
            else:
                direction = None
            if direction is not None:
                prompt_content = self.mv_prompts.get(f'prompt_{direction}', None)
            else:
                prompt_content = self.prompt
            if prompt_content:
                return prompt_content, self.cfg.prompt_template.format(prompt=prompt_content)

        return self.prompt, self.cfg.prompt_template.format(prompt=self.prompt)
    @torch.no_grad()
    def process_angle(
        self,
        angle: CameraAngle,
        verbose: bool = True
    ):
        """
        Perform the entire pipeline for a single angle (pitch, yaw).
        This method does NOT take the member variables as arguments;
        it uses them directly (self.cfg, self.obj_mesh, self.texture_dir, etc.).
        """
        yaw, pitch = angle.yaw, angle.pitch
        postfix = f"{yaw:.1f}_{pitch:.1f}"

        #Step1: Get rendering under current angle
        start = time.perf_counter()

        zoom, refine_type, ratio, mask, renderings = self.render_planar_projection(
            cfg=self.cfg,
            obj_mesh=self.mesh, 
            texture_dir=self.texture_dir,
            pitch=pitch,
            yaw=yaw,
            res=self.cfg.im_res,
            thresh=self.cfg.cos_thresh,
            device_idx=self.device_idx
        )

        log.info(f"Rendering took {time.perf_counter() - start:0.4f} seconds")


        if refine_type == RefineType.SKIP:
            return refine_type
        
        depth_map = self.pure_depth[(angle.yaw, angle.pitch)]
        # Create a subdirectory for this refinement step
        if verbose:
            refine_step_dir = self.data_path / f"refine_{postfix}_{zoom}"
            refine_step_dir.mkdir(parents=True, exist_ok=True)

            # Save inputs to image refiner for debugging/visualization
            # 1. Mask
            mask.save(refine_step_dir / "mask.png")
            
            # 2. Source (pure rendering)
            source_tensor = self.pure_renderings[(angle.yaw, angle.pitch)]
            source_pil = transforms.ToPILImage()(source_tensor)
            source_pil.save(refine_step_dir / "source.png")
            fg_mask = self.fg_masks[(angle.yaw, angle.pitch)]
            fg_mask.save(refine_step_dir / "fg_mask.png")


            # 3. Inpaint Image (current rendering)
            inpaint_pil = renderings["rgba"].convert("RGB")
            inpaint_pil.save(refine_step_dir / "inpaint.png")

            depth_map.save(refine_step_dir / "depth.png")

        #Step2: Refine the rendered image, given mask, rendered depth, re-rendering
        start = time.perf_counter()
        gen_prompt, edit_prompt = self.get_prompt(angle)
        refined_rgb = self.image_refiner(
            # source = transforms.ToTensor()(renderings["rgba"].convert("RGB")),
            # TODO: use depth? 
            source = [self.pure_renderings[(angle.yaw, angle.pitch)], transforms.ToTensor()(depth_map.convert("RGB"))],
            mask = transforms.ToTensor()(mask),
            prompt=edit_prompt,
            # inversion_prompt=gen_prompt,
            inversion_prompt=" ", #TODO: Ablate the prompt used for inversion
            negative_prompt=self.cfg.negative_prompt,
            inpaint_image = transforms.ToTensor()(renderings["rgba"].convert("RGB")),
            verbose = verbose
        )[0]
        log.info(f"Image Refinement took {time.perf_counter() - start:0.4f} seconds")
        
        # Save output to the same subdirectory
        if verbose:
            refined_rgb.save(refine_step_dir / "refined_output.png")
        torch.cuda.empty_cache()
        
        #Step3: Remove the background of the generated image
        start = time.perf_counter()
        refined_rgba = self.remove_bg(refined_rgb)
        log.info(f"Background Removal took {time.perf_counter() - start:0.4f} seconds")
        move_existing_files(
            self.texture_dir,
            self.data_path / "old_textures" if verbose else None,
            yaw,
            pitch,
            epsilon=1e-3
        )
        refined_rgba.save(self.texture_dir / f"refined_{postfix}_{zoom}.png")

        if refine_type == RefineType.NEGLIGIBLE:
            return refine_type


        #Step4: Monocular Normal Estimation (by Marigold)
        start = time.perf_counter()
        mari_pipe_out = self.normal_estimator(refined_rgb.convert("RGB"))
        mari_normal_colored = mari_pipe_out.normals_img
        log.info(f"Normal Estimation took {time.perf_counter() - start:0.4f} seconds")
        
        # Save predicted normal map
        if verbose:
            mari_normal_colored.save(refine_step_dir / "predicted_normal.png")


        torch.cuda.empty_cache()

        alpha_mask = refined_rgba.split()[-1]  # RGBA alpha
        # mask_path = self.remeshing_dir / "masks" / f"mask_{postfix}.png"
        # alpha_mask.save(mask_path)
        # print(f"Saved alpha mask to {mask_path}")

        torch.cuda.empty_cache()

        #Step5: Render depth prior
        start = time.perf_counter()
        
        depth_map, depth_mask, mv_mat = render_depth_prior(
            obj_path=self.current_obj_fp,
            im_res=self.cfg.im_res, 
            pitch=pitch, 
            yaw=yaw,
            zoom=zoom,
        )
        log.info(f"Depth Prior Rendering took {time.perf_counter() - start:0.4f} seconds")
        
        # Save depth prior mask
        if verbose:
            Image.fromarray((depth_mask * 255).astype(np.uint8)).save(refine_step_dir / "mask_depth_prior.png")
            Image.fromarray((depth_mask * 255).astype(np.uint8)).save(refine_step_dir / "mask_depth_prior.png")


        #Step6: Prepare normal maps for fusion
        start = time.perf_counter()
        blended_normal_map = load_and_blend_maps(
            base_map_input=renderings["normal"],
            detail_map_input=mari_normal_colored,
            l0_smoother=L0Smoothing()
        )
        log.info(f"Normal Map Fusion took {time.perf_counter() - start:0.4f} seconds")
        
        # Save blended normal map
        if verbose:
            blended_normal_map.save(refine_step_dir / "blended_normal.png")
        
        #Step7: Normal & depth prior fusion
        start = time.perf_counter()
        inpaint_mask_pil = mask
        inpaint_mask_np  = np.array(inpaint_mask_pil).astype(bool)

        bini_surface, wu_wv_mask = run_bilateral_integration(
            save_path=None, # self.bini_save_dir,
            normal_input=blended_normal_map,
            normal_mask=np.array(alpha_mask).astype(bool),
            depth_map=depth_map,
            depth_mask=depth_mask, # Rendered coarse prior
            mv_mat=mv_mat,
            yaw=yaw,
            pitch=pitch,
            zoom=zoom,
            im_res=self.cfg.im_res,
            inpaint_mask=inpaint_mask_np * np.array(alpha_mask).astype(bool), # Inpaint target mask && actual result fg mask
            depth_lambda=self.cfg.poisson.bini_params.depth_lambda,
            depth_lambda2=self.cfg.poisson.bini_params.depth_lambda2,
            k=self.cfg.poisson.bini_params.k,
            iters=self.cfg.poisson.bini_params.iters,
            tol=self.cfg.poisson.bini_params.tol,
            cgiter=self.cfg.poisson.bini_params.cgiter,
            cgtol=self.cfg.poisson.bini_params.cgtol,
            verbose = verbose
        )
        log.info(f"Normal & Depth Prior Fusion took {time.perf_counter() - start:0.4f} seconds")
        
        #Step8: Refind and postprocess 
        start = time.perf_counter()
        next_rgba_np = np.array(refined_rgba)
        next_rgba_np[..., -1] = next_rgba_np[..., -1] * wu_wv_mask
        next_rgba_wu_wv_mask = Image.fromarray(next_rgba_np)
        
        # Save to texture dir as final refined texture for this angle
        next_rgba_wu_wv_mask.save(self.texture_dir / f"refined_{postfix}_{zoom}.png")

        # Save the final mask produced by all checks (Depth Prior + BiNI confidence)
        if verbose:
            final_mask_pil = Image.fromarray((wu_wv_mask * 255).astype(np.uint8))
            final_mask_pil.save(refine_step_dir / "mask_final_accepted.png")
        
        # 9-1) bini postprocess
        bini_mesh = bini_surface.extract_surface().triangulate()
        bini_faces_as_array = bini_mesh.faces.reshape((bini_mesh.n_faces, 4))[:, 1:]
        bini_points_as_array = bini_mesh.points

        # Create a PyMeshLab MeshSet
        ms = pymeshlab.MeshSet()
        pm_mesh = pymeshlab.Mesh(vertex_matrix=bini_points_as_array, 
                                 face_matrix=bini_faces_as_array)
        ms.add_mesh(pm_mesh, "bini_mesh")

        # Remove small islands
        ms.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=pymeshlab.PercentageValue(5.),
            removeunref=True
        )
        
        # 5. Retrieve final mesh data
        final_mesh     = ms.current_mesh()
        final_vertices = final_mesh.vertex_matrix()
        final_faces    = final_mesh.face_matrix()

        # ms.save_current_mesh(
        #     str(self.bini_save_dir / f"bini_mesh_k_{self.cfg.poisson.bini_params.k}_lambda1_{self.cfg.poisson.bini_params.depth_lambda}_{yaw:.1f}_{pitch:.1f}.ply"),
        # )

        bini_mesh_torch = Mesh(
            device='cpu',
            v=torch.Tensor(final_vertices).float(),
            f=torch.Tensor(final_faces).int(),
            )

        # 10) Accumulate bini surfaces
        self.bini_surfaces.append(bini_mesh_torch)


        # 11) Poisson Pipeline: OBJ -> PLY -> Poisson -> Updated OBJ
        if not self.current_obj_fp.with_suffix('.ply').exists():
            ply_path = convert_obj_to_ply(self.current_obj_fp)
        else:
            ply_path = self.current_obj_fp.with_suffix('.ply')
    
        poisson_save_dir = self.data_path / "poisson"
        partial_meshes_dir = self.data_path / "partial_meshes"
        
        ply_path = process_poisson(
            angle,
            ply_path,
            self.bini_surfaces,
            self.texture_dir,
            None, # mask_path
            None, # normal_path
            poisson_save_dir,
            partial_meshes_dir,
            self.seen_angle_list,
            poisson_bin_fp=self.cfg.poisson.bin_fp,
            im_res=self.cfg.im_res,
            poisson_depth=self.cfg.poisson.poisson_depth,
            envelope_depth=self.cfg.poisson.envelope_depth,
            seen_thresh=self.cfg.poisson.bini_params.seen_thresh,
            cleanup=not verbose
        )

        updated_obj_path = convert_ply_to_obj(ply_path)

        if not verbose:
            if Path(ply_path).exists():
                try:
                    Path(ply_path).unlink()
                except OSError:
                    pass
            
            # Remove previous intermediate file if it exists and is not the original mesh
            if "poisson" in str(self.current_obj_fp) and self.current_obj_fp.exists():
                 try:
                     self.current_obj_fp.unlink()
                 except OSError:
                     pass
            
            # Move current result out of poisson dir
            new_location = self.data_path / "mesh_refined_current.obj"
            shutil.move(str(updated_obj_path), str(new_location))
            updated_obj_path = new_location
            
            # Try to remove poisson dir
            try:
                poisson_save_dir.rmdir()
            except OSError:
                pass

        self.current_obj_fp   = updated_obj_path
        self.mesh         = Obj.open(updated_obj_path)

        return refine_type

    def bake_mesh(self):
        log.info("Starting mesh baking...")
        bake_start = time.perf_counter()
        
        # Define output directory for textured mesh
        textured_mesh_dir = self.data_path / "output_textured_mesh"
        textured_mesh_dir.mkdir(parents=True, exist_ok=True)

        # Create a PyMeshLab MeshSet
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(self.current_obj_fp))
        
        # Decimate
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=100_000,
            targetperc=0.,
            preserveboundary=True,
            preservenormal=True,
            optimalplacement=True,
            qualitythr=1.0,
            autoclean=True
        )
        
        # Save decimated mesh
        decimated_mesh_path = self.data_path / "final_deci_mesh.obj"
        ms.save_current_mesh(str(decimated_mesh_path))
        
        # Unwrap UVs
        unwrapped_obj_path = HeadlessBaker.unwrap_mesh_with_xatlas(decimated_mesh_path, textured_mesh_dir)
        unwrapped_obj = Obj.open(unwrapped_obj_path)

        
        baker = HeadlessBaker(
            self.cfg.projection.bake_vertex_shader_path,
            unwrapped_obj,
            self.texture_dir,
            self.device_idx,
        )
        
        # Bake Texture
        # Assuming 4K texture resolution
        baked_image = baker.bake_texture(uv_texture_size=(1024 * 4, 1024 * 4))
        baked_image = baker.pad_uvs_cupy(baked_image)
        baked_image_path = textured_mesh_dir / "baked_texture_map.png"
        baked_image.save(baked_image_path)

        # Assign Texture
        baker.assign_baked_texture_to_mesh(
            obj_path=unwrapped_obj_path,
            baked_texture_filename=baked_image_path,
        )

        bake_end = time.perf_counter()
        log.info(f"Mesh baking took {bake_end - bake_start:0.4f} seconds")

        return unwrapped_obj_path

    @torch.no_grad()
    def __call__(
        self,
        data_path: str,
        verbose: bool= True
        ):
        """
        Assume the data directory structure
        data_path 
           |-- textures
           |-- prompt.txt or prompts_mv.json
           |-- mesh.obj
           |-- mesh.mtl
        """
        self.data_path = Path(data_path)
        self.raw_texture_dir = self.data_path / 'textures'
        self.texture_dir = self.data_path / 'textures_on'
        start = time.perf_counter()
        if self.texture_dir.exists():
            for f in self.texture_dir.glob("*.png"):
                try:
                    f.unlink()
                except OSError:
                    pass
        else:
            self.texture_dir.mkdir(parents=True, exist_ok=True)
        base_schedule = self.cfg.camera_schedule or []
        num_angles = len(base_schedule)
        copy_texture_files(self.raw_texture_dir, self.texture_dir, num_angles)
        log.info(f'Copying textures took {time.perf_counter() - start:0.4f} seconds')
        self.current_obj_fp = self.data_path / 'mesh_processed.obj'
        self.mesh = Obj.open(self.current_obj_fp)
        
        # Create old_textures directory once
        if verbose:
            (self.data_path / "old_textures").mkdir(parents=True, exist_ok=True)
        
        # Initialize state lists
        self.seen_angle_list = []
        self.bini_surfaces = []

        mv_prompts_path = self.data_path / 'prompts_mv.json'

        prompt_path = self.data_path / "prompt.txt"

        # Load global prompt first (as fallback)
        if prompt_path.exists():
            with open(prompt_path, 'r') as f:
                prompt = f.read().strip()
            self.prompt = self.cfg.prompt_template.format(prompt=prompt)
        else:
            self.prompt = self.cfg.prompt_template.format(prompt="")

        # Load view dependent prompts if available
        if mv_prompts_path.exists():
            with open(mv_prompts_path, 'r') as f:
                self.mv_prompts = json.load(f)
            if not self.cfg.enable_mv_prompts:
                log.warning(f'View dependent prompts are loaded from {mv_prompts_path}, but disabled in config')
            log.info(f'Loaded view dependent prompts from {mv_prompts_path}')
        else:
            log.info(f'view dependent prompts not found, use global prompt: {self.prompt}') 

        # Pure images along the camera schedule, to avoid mixed-refined images
        # during the refinement process jump out the distribution of conditioning mechanism
        self.pure_renderings = {}
        self.fg_masks = {}
        self.pure_depth = {}
        for entry in self.camera_schedule:
            camera = CameraAngle(yaw=entry['yaw'] % 360, pitch=entry['pitch'])
            yaw, pitch = camera.yaw, camera.pitch
            _, _, _, _, ret = self.render_planar_projection(
                cfg=self.cfg,
                obj_mesh=self.mesh,
                texture_dir=self.texture_dir,
                pitch=pitch,
                yaw=yaw,
                res=self.cfg.im_res,
                thresh=self.cfg.cos_thresh,
                device_idx=self.device_idx
            )
            self.fg_masks[(yaw, pitch)] = ret['rgba'].split()[-1].point(lambda p: 255 if p > 0 else 0)
            self.pure_renderings[(yaw, pitch)] = transforms.ToTensor()(ret['rgba'].convert("RGB"))
            self.pure_depth[(yaw, pitch)] = ret['depth']



        # Start iteration 
        for i, entry in enumerate(self.camera_schedule):
            camera = CameraAngle(yaw=entry['yaw'] % 360, pitch=entry['pitch'])
            refine_type = self.process_angle(camera, verbose=verbose)
            if refine_type != RefineType.SKIP:
                self.seen_angle_list.append(camera)        
        
        
        if self.cfg.bake:
            self.bake_mesh()