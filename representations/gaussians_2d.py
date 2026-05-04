import torch 
import numpy as np 
from torch import nn 
import os 
from plyfile import PlyData, PlyElement 
import math 
import utils3d
from representations.utils.sh import RGB2SH, SH2RGB, eval_sh
from simple_knn._C import distCUDA2 
from representations.utils.cam import yaw_pitch_r_fov_to_extrinsics_intrinsics, extr_intr_to_viewpoint_cam
from utils.registry import RENDERER_REGISTRY
from utils.config import BaseConfig 
from dataclasses import dataclass, field
from PIL import Image 
import cv2
import copy
import open3d.core as o3c 
import open3d.t.geometry as tgeom
import xatlas 
import nvdiffrast.torch as dr
from tqdm import tqdm 
from typing import List, Literal 
import open3d as o3d 
import trimesh 
from representations.utils.gaussians import build_rotation, build_scaling_rotation, get_expon_lr_func, inverse_sigmoid, BasicPointCloud, depth_to_normal
try:
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
except:
    print(f'Please install diff-surfel-rasterization (https://github.com/hbb1/diff-surfel-rasterization) if you want to use 2D Gaussian Splatting')
from utils.logger import setup_logging, get_logger
setup_logging()
log = get_logger(__name__)
@dataclass
class Gaussian2DConfig(BaseConfig):
    aabb: list = field(default_factory=lambda: [-0.5, -0.5, -0.5, 1.0, 1.0, 1.0])
    sh_degree: int = 0
    minimum_kernel_size: float = 0.01
    scaling_bias: float = 0.01
    opacity_bias: float = 0.1

    position_lr_init: float = 0.001 
    position_lr_final: float = 0.0002
    position_lr_delay_mult: float = 0.02
    position_lr_max_steps: int = 700
    feature_lr: float = 0.01
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.005
    percent_dense: float = 0.01 
    density_start_iter: int = 100
    density_end_iter: int = 500
    densification_interval: int = 100 
    opacity_reset_interval: int = 100000
    densify_grad_threshold: float = 0.01

    # This is moved to this representation for convenience to build a general ScoreDistillationSampling 
    # regardless of representations adopted 
    lambda_normal: float = 0.1 
    lambda_dist: float = 0.01 
    normal_step: int = 100 
    dist_step: int = 300 

    

class GaussianModel:

    Config = Gaussian2DConfig
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation, device=self.device).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device=self.device)
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        self.scale_bias = self.scaling_inverse_activation(torch.tensor(self.scaling_bias)).to(self.device)
        self.rots_bias = torch.zeros((4)).to(self.device)
        self.rots_bias[0] = 1
        self.opacity_bias = self.inverse_opacity_activation(torch.tensor(self.opacity_bias)).to(self.device)


    def __init__(
        self, 
        config: Gaussian2DConfig,
        device: str="cuda"
    ):
        self.device = device 
        self.cfg = config
        self.aabb = torch.tensor(config.aabb, dtype=torch.float32, device=device)
        self.active_sh_degree = 0
        self.minimum_kernel_size = config.minimum_kernel_size
        self.scaling_bias = config.scaling_bias
        self.opacity_bias = config.opacity_bias
        self.max_sh_degree = config.sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
    
    def to(self, device):
        self.device = device 
        self.aabb = self.aabb.to(device)
        self._xyz = self._xyz.to(device)
        self._features_dc = self._features_dc.to(device)
        self._features_rest = self._features_rest.to(device)
        self._scaling = self._scaling.to(device)
        self._rotation = self._rotation.to(device)
        self._opacity = self._opacity.to(device)
        self.max_radii2D = self.max_radii2D.to(device)
        self.xyz_gradient_accum = self.xyz_gradient_accum.to(device)
        self.denom = self.denom.to(device)
        self.scale_bias = self.scale_bias.to(device)
        self.rots_bias = self.rots_bias.to(device)
        self.opacity_bias = self.opacity_bias.to(device)
        return self
    
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        scales = self.scaling_activation(self._scaling + self.scale_bias)
        scales = torch.square(scales) + self.minimum_kernel_size ** 2
        scales = torch.sqrt(scales)
        return scales
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation + self.rots_bias[None, :])
    
    @property
    def get_xyz(self):
        return self._xyz * self.aabb[None, 3:] + self.aabb[None, :3]
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity + self.opacity_bias)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation + self.rots_bias[None, :])

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    def update(self, step, out):
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.update_learning_rate(step)
        torch.cuda.synchronize()
        if step >= self.cfg.density_start_iter and step <= self.cfg.density_end_iter:
            viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
            if self.max_radii2D.shape[0] != visibility_filter.shape[0]:
                self.max_radii2D = torch.zeros((visibility_filter.shape[0]), device=visibility_filter.device)
            self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter], radii[visibility_filter])
            self.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if step % self.cfg.densification_interval == 0:
                log.info(f'[Step {step}] Before Densification, number of points: {self.get_xyz.shape[0]}')
                self.densify_and_prune(self.cfg.densify_grad_threshold, min_opacity=0.01, extent=4, max_screen_size=1)
                log.info(f'[Step {step}] After Densification, number of points: {self.get_xyz.shape[0]}')
            if step % self.cfg.opacity_reset_interval == 0:
                self.reset_opacity()

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float = 1, transform=[[1, 0, 0], [0, 0, -1], [0, 1, 0]]):
        self.spatial_lr_scale = spatial_lr_scale
        xyz = np.asarray(pcd.points)
        if transform is not None:
            transform = np.array(transform)
            xyz = xyz @ transform

        aabb_min = self.aabb[:3].cpu().numpy()      
        aabb_size = self.aabb[3:].cpu().numpy()     
        xyz_normalized = (xyz - aabb_min) / aabb_size 

        fused_point_cloud = torch.tensor(xyz_normalized, dtype=torch.float32, device=self.device)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors), dtype=torch.float32, device=self.device))

        features = torch.zeros(
            (fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2),
            dtype=torch.float32, device=self.device
        )
        features[:, :3, 0] = fused_color


        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(xyz_normalized).float().to(self.device)),
            1e-7
        )
        
        scales_raw = torch.sqrt(dist2)  
        scales_hidden = self.scaling_inverse_activation(
            torch.clamp_min(scales_raw, self.minimum_kernel_size)
        ) - self.scale_bias
        scales_hidden = scales_hidden[..., None].repeat(1, 2)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
        rots[:, 0] = 1.0 
        rots_hidden = rots - self.rots_bias[None, :]  


        opacity_raw = 0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float32, device=self.device)
        opacity_hidden = self.inverse_opacity_activation(opacity_raw) - self.opacity_bias

        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales_hidden.requires_grad_(True))
        self._rotation = nn.Parameter(rots_hidden.requires_grad_(True))
        self._opacity = nn.Parameter(opacity_hidden.requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0],), device=self.device)

    def create_from_3dgs(
            self, 
            ply_path, 
            num_pts = None,
            transform=[[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        ):
        plydata = PlyData.read(ply_path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        if num_pts is not None and num_pts < xyz.shape[0]:
            indices = np.random.choice(xyz.shape[0], num_pts, replace=False)
            xyz = xyz[indices]
            opacities = opacities[indices]
            features_dc = features_dc[indices]
            features_extra = features_extra[indices]

        if transform is not None:
            transform = np.array(transform)
            xyz = np.matmul(xyz, transform)

        aabb_min = self.aabb[:3].cpu().numpy()
        aabb_size = self.aabb[3:].cpu().numpy()
        xyz_normalized = (xyz - aabb_min) / aabb_size

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(xyz_normalized).float().to(self.device)),
            1e-7
        )
        
        scales_raw = torch.sqrt(dist2)  
        scales_hidden = self.scaling_inverse_activation(
            torch.clamp_min(scales_raw, self.minimum_kernel_size)
        ) - self.scale_bias
        scales_hidden = scales_hidden[..., None].repeat(1, 2)

        rots = torch.zeros((xyz.shape[0], 4), device=self.device)
        rots[:, 0] = 1.0  # identity quat
        rots_hidden = rots - self.rots_bias[None, :]

        self._xyz = nn.Parameter(torch.tensor(xyz_normalized, dtype=torch.float, device=self.device).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=self.device).requires_grad_(True))
        self._scaling = nn.Parameter(scales_hidden.requires_grad_(True))
        self._rotation = nn.Parameter(rots_hidden.requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree
    
    def training_setup(self):
        self.percent_dense = self.cfg.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)

        l = [
            {'params': [self._xyz], 'lr': self.cfg.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': self.cfg.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': self.cfg.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': self.cfg.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': self.cfg.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': self.cfg.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=self.cfg.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=self.cfg.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=self.cfg.position_lr_delay_mult,
                                                    max_steps=self.cfg.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save(self, path):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        opacities = self.get_opacity.detach().cpu().numpy()
        scales    = self.get_scaling.detach().cpu().numpy()
        rotations = self.get_rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    def load_ply(
        self,
        path,
        transform=[[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        num_pts=None
    ):
        plydata = PlyData.read(path)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        if len(scale_names) == 3:
            return self.create_from_3dgs(path, num_pts=num_pts, transform=transform)
        elif len(scale_names) > 2: 
            scale_names = scale_names[:2]
        
        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"])
        ), axis=1)

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]


        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # Rotations
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        if num_pts is not None and num_pts > 0 and num_pts < xyz.shape[0]:
            indices = np.random.choice(xyz.shape[0], num_pts, replace=False)
            xyz = xyz[indices]
            opacities = opacities[indices]
            features_dc = features_dc[indices]
            features_extra = features_extra[indices]
            scales = scales[indices]
            rots = rots[indices]

        # Apply coordinate + rotation transform if provided
        if transform is not None:
            transform = np.array(transform)
            xyz = np.matmul(xyz, transform)
            rotation_matrices = utils3d.numpy.quaternion_to_matrix(rots)
            rotation_matrices = np.matmul(transform.T, rotation_matrices)
            rots = utils3d.numpy.matrix_to_quaternion(rotation_matrices)

        xyz_t         = torch.from_numpy(xyz).float().to(self.device)
        opacities_act = torch.from_numpy(opacities).float().to(self.device)     
        scales_act    = torch.from_numpy(scales).float().to(self.device)       
        rots_act      = torch.from_numpy(rots).float().to(self.device)        
        features_dc_t   = torch.from_numpy(features_dc).float().to(self.device)
        features_extra_t = torch.from_numpy(features_extra).float().to(self.device)

        opacity_hidden = self.inverse_opacity_activation(opacities_act) - self.opacity_bias

        scales_clamped = torch.clamp(scales_act ** 2 - self.minimum_kernel_size ** 2, min=1e-8)
        scaling_hidden = self.scaling_inverse_activation(torch.sqrt(scales_clamped)) - self.scale_bias

        rotation_hidden = rots_act - self.rots_bias[None, :]

        xyz_hidden = (xyz_t - self.aabb[None, :3]) / self.aabb[None, 3:]

        self._xyz = nn.Parameter(xyz_hidden.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc_t.transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features_extra_t.transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(opacity_hidden.requires_grad_(True))
        self._scaling = nn.Parameter(scaling_hidden.requires_grad_(True))
        self._rotation = nn.Parameter(rotation_hidden.requires_grad_(True))

        self.max_radii2D = torch.zeros(xyz_hidden.shape[0], device=self.device)
        self.active_sh_degree = self.max_sh_degree


    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        opacities_new = opacities_new - self.opacity_bias
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune(self, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        
        # Ensure max_radii2D matches current points
        if self.max_radii2D.shape[0] != self.get_xyz.shape[0]:
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # Gradients may be None if no backward pass reached the screenspace points.
        grad = viewspace_point_tensor.grad
        self.xyz_gradient_accum[update_filter] += torch.norm(grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


@dataclass
class Gaussian2DRendererConfig:
    znear: float = 0.8
    zfar: float = 1.6
    radius: float = 0.5
    num_pts: int = 5000
    transform: list = field(default_factory=lambda: [[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    bg_color: list = field(default_factory=lambda: [0.5, 0.5, 0.5])
    gaussian_config: Gaussian2DConfig = Gaussian2DConfig()
    type: str = 'Gaussian2d'

    # Noise deviation added after loading the checkpoint
    xyz_std: float = 0.0
    opac_std: float = 0.0
    scale_std: float = 0.0
    rotate_std: float = 0.0
    feat_std: float = 0.0

    # For mesh extraction
    mesh_bg_color: tuple = field(default_factory=lambda: (0, 0, 0))
    voxel_size: float = 0.004
    sdf_trunc: float = 0.02
    depth_trunc: float = 3.0
    bake: bool = True
    bake_mode: Literal['fast', 'opt'] = 'opt'
    texture_size: int = 2048
    max_clusters_to_keep: int = 1000
    min_triangles: int = 100
    min_area_ratio: float = 0.001
    smooth_iterations: int = 2
    smooth_lambda: float = 0.5
    yaws_num: int = 36
    pitchs: list = field(default_factory=lambda: [-45, -20, 0, 20, 45])
    reconstruct_radius: float = 1.7
    reconstruct_fov: float = 49.1


@RENDERER_REGISTRY.register('Gaussian2d')
class Renderer:
    Config = Gaussian2DRendererConfig
    def __init__(
        self, 
        cfg: Gaussian2DRendererConfig,
        device='cuda'
    ):
        self.device = device
        self.representation = GaussianModel(Gaussian2DConfig(**cfg.gaussian_config), device=device)
        self.cfg = cfg
        self.bg_color = torch.tensor(
            cfg.bg_color,
            dtype=torch.float32,
            device=device,
        )

    def initialize(self, input=None):
        if input is None:
            phis = np.random.random((self.cfg.num_pts,)) * 2 * np.pi
            costheta = np.random.random((self.cfg.num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((self.cfg.num_pts,))
            radius = self.cfg.radius * np.cbrt(mu)
            x = radius * np.sin(thetas) * np.cos(phis)
            y = radius * np.sin(thetas) * np.sin(phis)
            z = radius * np.cos(thetas)
            xyz = np.stack((x, y, z), axis=1)

            shs = np.random.random((self.cfg.num_pts, 3)) / 255.0
            pcd = BasicPointCloud(
                points=xyz, colors=SH2RGB(shs), normals=np.zeros((self.cfg.num_pts, 3))
            )
            self.representation.create_from_pcd(pcd, 10)
        elif isinstance(input, BasicPointCloud):
            self.representation.create_from_pcd(input, 1)
        else:
            self.representation.load_ply(input, num_pts=self.cfg.num_pts, transform=self.cfg.transform)
        
        with torch.no_grad():
            self.representation._xyz.add_(torch.randn_like(self.representation._xyz) * self.cfg.xyz_std)
            self.representation._opacity.add_(torch.randn_like(self.representation._opacity) * self.cfg.opac_std)
            self.representation._scaling.add_(torch.randn_like(self.representation._scaling) * self.cfg.scale_std)
            self.representation._rotation.add_(torch.randn_like(self.representation._rotation) * self.cfg.rotate_std)
            self.representation._features_dc.add_(torch.randn_like(self.representation._features_dc) * self.cfg.feat_std)
        
        
    
    def training_setup(self):
        self.representation.training_setup()
        self.representation.active_sh_degree = self.representation.max_sh_degree

    def render(
        self,
        azimuth,
        elevation,
        radius, 
        fov, 
        depth_ratio = 0.0, 
        scaling_modifier=1.0,
        width=512,
        height=512,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
    ):
        extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics(azimuth, elevation, radius, fov, device=self.device)
        viewpoint_camera = extr_intr_to_viewpoint_cam(extr, intr, width, height, self.cfg.znear, self.cfg.zfar)
        screenspace_points = (
            torch.zeros_like(
                self.representation.get_xyz,
                dtype=self.representation.get_xyz.dtype,
                requires_grad=True,
                device=self.device,
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        bg_color = self.cfg.bg_color if bg_color is None else torch.tensor(bg_color, dtype=torch.float32, device=self.device)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.representation.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        means3D = self.representation.get_xyz
        means2D = screenspace_points
        opacity = self.representation.get_opacity

        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            splat2world = self.representation.get_covariance(scaling_modifier)
            W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
            near, far = viewpoint_camera.znear, viewpoint_camera.zfar
            ndc2pix = torch.tensor([
                [W / 2, 0, 0, (W-1) / 2],
                [0, H / 2, 0, (H-1) / 2],
                [0, 0, far-near, near],
                [0, 0, 0, 1]]).float().to(self.device).T
            world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
            cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) 
        else:
            scales = self.representation.get_scaling
            rotations = self.representation.get_rotation
        

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if convert_SHs_python:
                shs_view = self.representation.get_features.transpose(1, 2).view(
                    -1, 3, (self.representation.max_sh_degree + 1) ** 2
                )
                dir_pp = self.representation.get_xyz - viewpoint_camera.camera_center.repeat(
                    self.representation.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.representation.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.representation.get_features
        else:
            colors_precomp = override_color


        rendered_image, radii, allmap = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp
        )

        rets =  {
            "render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii
            }

        render_alpha = allmap[1:2]

        render_normal = allmap[2:5]
        render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
        
        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

        render_depth_expected = allmap[0:1]
        render_depth_expected = (render_depth_expected / render_alpha)
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
        
        render_dist = allmap[6:7]

        surf_depth = render_depth_expected * (1-depth_ratio) + (depth_ratio) * render_depth_median
        
        surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
        surf_normal = surf_normal.permute(2,0,1)
        surf_normal = surf_normal * (render_alpha).detach()


        rets.update({
                'rend_alpha': render_alpha,
                'rend_normal': render_normal,
                'rend_dist': render_dist,
                'surf_depth': surf_depth,
                'surf_normal': surf_normal,
        })

        return rets

    def get_reg(self, out, step, **kwargs):
        """
        Representation specific regularization term
        """
        lambda_normal = self.representation.cfg.lambda_normal if step >= self.representation.cfg.normal_step else 0.0
        lambda_dist = self.representation.cfg.lambda_dist if step >= self.representation.cfg.dist_step else 0.0
        rend_dist = out['rend_dist']
        rend_normal = out['rend_normal']
        surf_normal = out['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).sum()
        dist_loss = lambda_dist * (rend_dist).sum()
        return normal_loss + dist_loss
    
    def to(self, device):
        self.device = device 
        self.representation.to(device)
        self.bg_color = self.bg_color.to(device)
        return self
    
    @torch.no_grad()
    def render_video(self, num_frames=300, r=3.0, fov=40, bg_color=None):
        yaws = torch.linspace(-180, 180, num_frames)
        yaws = yaws.tolist() 
        images = []
        for yaw in yaws:
            rendering = self.render(
                        yaw, 5, r, fov, bg_color = bg_color
                    )["render"]
            images.append(np.clip(rendering.detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
        return images

    def extract_mesh(self, verbose: bool = True):
        extractor = MeshExtractor(bg_color=self.cfg.mesh_bg_color, device=self.device)
        yaws = []
        pitchs = []
        for p in self.cfg.pitchs:
            for i in range(self.cfg.yaws_num):
                yaws.append(i * 360 / self.cfg.yaws_num)
                pitchs.append(p)
        extractor.reconstruct(
            renderer=self,
            yaws=yaws,
            pitchs=pitchs,
            radius=self.cfg.reconstruct_radius,
            fov=self.cfg.reconstruct_fov,
        )
        mesh = extractor.extract_mesh(
            voxel_size=self.cfg.voxel_size,
            sdf_trunc=self.cfg.sdf_trunc,
            depth_trunc=self.cfg.depth_trunc,
            bake=self.cfg.bake,
            bake_mode=self.cfg.bake_mode,
            texture_size=self.cfg.texture_size,
            verbose=verbose,
            max_clusters_to_keep=self.cfg.max_clusters_to_keep,
            min_triangles=self.cfg.min_triangles,
            min_area_ratio=self.cfg.min_area_ratio,
            smooth_iterations=self.cfg.smooth_iterations,
            smooth_lambda=self.cfg.smooth_lambda
        )
        return mesh 


class MeshExtractor:
    def __init__(self, bg_color = [0, 0, 0], device='cuda'):
        self.device = device
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=device)
    
    @staticmethod
    def bake_texture(
        vertices: np.array,
        faces: np.array,
        uvs: np.array,
        observations: List[np.array],
        masks: List[np.array],
        extrinsics: List[np.array],
        intrinsics: List[np.array],
        texture_size: int = 2048,
        near: float = 0.1,
        far: float = 10.0,
        mode: Literal['fast', 'opt'] = 'opt',
        lambda_tv: float = 1e-2,
        verbose: bool = False,
        device: torch.device = torch.device('cuda')
    ):
        """
        Bake texture to a mesh from multiple observations.

        Args:
            vertices (np.array): Vertices of the mesh. Shape (V, 3).
            faces (np.array): Faces of the mesh. Shape (F, 3).
            uvs (np.array): UV coordinates of the mesh. Shape (V, 2).
            observations (List[np.array]): List of observations. Each observation is a 2D image. Shape (H, W, 3).
            masks (List[np.array]): List of masks. Each mask is a 2D image. Shape (H, W).
            extrinsics (List[np.array]): List of extrinsics. Shape (4, 4).
            intrinsics (List[np.array]): List of intrinsics. Shape (3, 3).
            texture_size (int): Size of the texture.
            near (float): Near plane of the camera.
            far (float): Far plane of the camera.
            mode (Literal['fast', 'opt']): Mode of texture baking.
            lambda_tv (float): Weight of total variation loss in optimization.
            verbose (bool): Whether to print progress.
        """
        vertices = torch.tensor(vertices).to(device)
        faces = torch.tensor(faces.astype(np.int32)).to(device)
        uvs = torch.tensor(uvs).to(device)
        observations = [torch.tensor(obs / 255.0).float().to(device) for obs in observations]
        masks = [torch.tensor(m>0).bool().to(device) for m in masks]
        views = [utils3d.torch.extrinsics_to_view(torch.tensor(extr).to(device)) for extr in extrinsics]
        projections = [utils3d.torch.intrinsics_to_perspective(torch.tensor(intr).to(device), near, far) for intr in intrinsics]

        if mode == 'fast':
            texture = torch.zeros((texture_size * texture_size, 3), dtype=torch.float32).to(device)
            texture_weights = torch.zeros((texture_size * texture_size), dtype=torch.float32).to(device)
            rastctx = utils3d.torch.RastContext(backend='cuda') # This uses default cuda, might need update if utils3d supports device
            for observation, view, projection in tqdm(zip(observations, views, projections), total=len(observations), disable=not verbose, desc='Texture baking (fast)'):
                with torch.no_grad():
                    rast = utils3d.torch.rasterize_triangle_faces(
                        rastctx, vertices[None], faces, observation.shape[1], observation.shape[0], uv=uvs[None], view=view, projection=projection
                    )
                    uv_map = rast['uv'][0].detach().flip(0)
                    mask = rast['mask'][0].detach().bool() & masks[0]
                
                # nearest neighbor interpolation
                uv_map = (uv_map * texture_size).floor().long()
                obs = observation[mask]
                uv_map = uv_map[mask]
                idx = uv_map[:, 0] + (texture_size - uv_map[:, 1] - 1) * texture_size
                texture = texture.scatter_add(0, idx.view(-1, 1).expand(-1, 3), obs)
                texture_weights = texture_weights.scatter_add(0, idx, torch.ones((obs.shape[0]), dtype=torch.float32, device=texture.device))

            mask = texture_weights > 0
            texture[mask] /= texture_weights[mask][:, None]
            texture = np.clip(texture.reshape(texture_size, texture_size, 3).cpu().numpy() * 255, 0, 255).astype(np.uint8)

            # inpaint
            mask = (texture_weights == 0).cpu().numpy().astype(np.uint8).reshape(texture_size, texture_size)
            texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)

        elif mode == 'opt':
            rastctx = utils3d.torch.RastContext(backend='cuda')
            observations = [observations.flip(0) for observations in observations]
            masks = [m.flip(0) for m in masks]
            _uv = []
            _uv_dr = []
            for observation, view, projection in tqdm(zip(observations, views, projections), total=len(views), disable=not verbose, desc='Texture baking (opt): UV'):
                with torch.no_grad():
                    rast = utils3d.torch.rasterize_triangle_faces(
                        rastctx, vertices[None], faces, observation.shape[1], observation.shape[0], uv=uvs[None], view=view, projection=projection
                    )
                    _uv.append(rast['uv'].detach())
                    _uv_dr.append(rast['uv_dr'].detach())

            texture = torch.nn.Parameter(torch.zeros((1, texture_size, texture_size, 3), dtype=torch.float32).to(device))
            optimizer = torch.optim.Adam([texture], betas=(0.5, 0.9), lr=1e-2)

            def exp_anealing(optimizer, step, total_steps, start_lr, end_lr):
                return start_lr * (end_lr / start_lr) ** (step / total_steps)

            def cosine_anealing(optimizer, step, total_steps, start_lr, end_lr):
                return end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(np.pi * step / total_steps))
            
            def tv_loss(texture):
                return torch.nn.functional.l1_loss(texture[:, :-1, :, :], texture[:, 1:, :, :]) + \
                    torch.nn.functional.l1_loss(texture[:, :, :-1, :], texture[:, :, 1:, :])
        
            total_steps = 2500
            with tqdm(total=total_steps, disable=not verbose, desc='Texture baking (opt): optimizing') as pbar:
                for step in range(total_steps):
                    optimizer.zero_grad()
                    selected = np.random.randint(0, len(views))
                    uv, uv_dr, observation, mask = _uv[selected], _uv_dr[selected], observations[selected], masks[selected]
                    render = dr.texture(texture, uv, uv_dr)[0]
                    loss = torch.nn.functional.l1_loss(render[mask], observation[mask])
                    if lambda_tv > 0:
                        loss += lambda_tv * tv_loss(texture)
                    loss.backward()
                    optimizer.step()
                    # annealing
                    optimizer.param_groups[0]['lr'] = cosine_anealing(optimizer, step, total_steps, 1e-2, 1e-5)
                    pbar.set_postfix({'loss': loss.item()})
                    pbar.update()
            texture = np.clip(texture[0].flip(0).detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
            mask = 1 - utils3d.torch.rasterize_triangle_faces(
                rastctx, (uvs * 2 - 1)[None], faces, texture_size, texture_size
            )['mask'][0].detach().cpu().numpy().astype(np.uint8)
            texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)
        else:
            raise ValueError(f'Unknown mode: {mode}')

        return texture

    def clean(self):
        self.depthmaps = []
        self.rgbmaps = []
        self.extrinsics = []
        self.intrinsics = []
        self.yaws = []
        self.pitchs = []
    
    @torch.no_grad()
    def reconstruct(
        self,
        renderer,
        yaws,
        pitchs, 
        radius, 
        fov,
        height=1024,
        width=1024
    ):
        """
        Note that this function assume orbit camera looking at the origin
        """
        self.clean()
        self.yaws = yaws
        self.pitchs = pitchs
        for y, p in zip(yaws, pitchs):
            out = renderer.render(
                y, p, radius, fov, bg_color=self.background, height=height, width=width
            )
            rgb = (
                out["render"]
                .clamp(0.0, 1.0)
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
            depth = out["surf_depth"].clone()
            self.rgbmaps.append((rgb * 255.0).astype(np.uint8))
            self.depthmaps.append(np.ascontiguousarray(depth[0].cpu().numpy().astype(np.float32)))
            
        self.radius = radius
        self.extrinsics, self.intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
            yaws, pitchs, radius, fov
        )
        self.height, self.width = height, width
    
    def extract_mesh(
        self,
        voxel_size: float = 0.004,
        sdf_trunc: float = 0.02,
        depth_trunc: float = 3.0,
        bake: bool = True,
        bake_mode: Literal['fast', 'opt'] = 'opt',
        texture_size: int = 2048, 
        verbose: bool = False,
        max_clusters_to_keep: int = 1000,
        min_triangles: int = 100,
        min_area_ratio: float = 0.001,  
        smooth_iterations: int = 2,     
        smooth_lambda: float = 0.5
    ):
        """
        If bake is set to True, renderings are directly converted to texture map
        instead of vertices color
        """
        device_str = str(self.device)
        if ':' in device_str:
            type_str, index_str = device_str.split(':')
            o3d_device = o3c.Device(type_str, int(index_str))
        else:
            o3d_device = o3c.Device(device_str)

        trunc_voxel_multiplier = sdf_trunc / voxel_size

        vbg = tgeom.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1,), (1,), (3,)),
            voxel_size=voxel_size,
            block_resolution=16,
            block_count=200000,  # adjust if needed
            device=o3d_device
        )

        for i, (extrinsic, intrinsic) in enumerate(zip(self.extrinsics, self.intrinsics)):
            color   = self.rgbmaps[i]
            depth = self.depthmaps[i]

            depth = o3d.t.geometry.Image(depth)
            depth = depth.to(o3d_device)
            depth = depth.to(o3c.float32)
            color = o3d.t.geometry.Image(color)
            color = color.to(o3d_device)
            color = color.to(o3c.float32)

            extrinsic, intrinsic = self.extrinsics[i].clone(), self.intrinsics[i].clone()
            intrinsic[..., 0, :] *= self.width
            intrinsic[..., 1, :] *= self.height
            intrinsic = o3d.core.Tensor(intrinsic.cpu().numpy(), dtype=o3c.float64)
            extrinsic = o3d.core.Tensor(extrinsic.cpu().numpy(), dtype=o3c.float64)

            frustum_block_coords = vbg.compute_unique_block_coordinates(
                depth, 
                intrinsic,
                extrinsic, 
                1.0, depth_trunc
            )

            vbg.integrate(
                frustum_block_coords, 
                depth, 
                color,
                intrinsic,
                extrinsic,  
                1.0, depth_trunc
            )

        mesh = vbg.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        mesh = mesh.to_legacy()
        mesh = self.postprocess_mesh(
            mesh,
            max_clusters_to_keep,
            min_triangles,
            min_area_ratio,
            smooth_iterations,
            smooth_lambda,
        )
        if not bake:
            return mesh 
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.int32)
        
        vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
        vertices = vertices[vmapping]
        faces = indices

        masks = [np.any(image > 0, axis=-1) for image in self.rgbmaps]
        extrinsics = [extrinsic.cpu().numpy() for extrinsic in self.extrinsics]
        intrinsics = [intrinsic.cpu().numpy() for intrinsic in self.intrinsics]
        texture = self.bake_texture(
            vertices, faces, uvs, self.rgbmaps, masks, extrinsics, intrinsics, texture_size=texture_size, mode=bake_mode, verbose=verbose, device=self.device
        )

        texture = Image.fromarray(texture)
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=texture,
            roughnessFactor=1.0,
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8)
        )
        visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
        textured_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            visual=visual,
            process=False
        )
        return textured_mesh 


    def postprocess_mesh(
        self, 
        mesh, 
        max_clusters_to_keep: int = 1000,
        min_triangles: int = 100,
        min_area_ratio: float = 0.001,  
        smooth_iterations: int = 2,    
        smooth_lambda: float = 0.5
    ):
        """
        Post-process a mesh:
        - Keep the largest N connected components (clusters)
        - Remove very small clusters (floaters)
        - Optional: Laplacian smoothing for better surface quality
        - Remove degenerate geometry
        """
        mesh_clean = copy.deepcopy(mesh)

        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            triangle_clusters, cluster_n_triangles, cluster_areas = \
                mesh_clean.cluster_connected_triangles()

        triangle_clusters = np.asarray(triangle_clusters).squeeze()
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_areas = np.asarray(cluster_areas)

        if min_area_ratio > 0:
            total_area = cluster_areas.sum()
            min_area = total_area * min_area_ratio
            large_enough = cluster_areas >= min_area
        else:
            large_enough = cluster_n_triangles >= min_triangles

        sorted_indices = np.argsort(-cluster_n_triangles)
        n_to_keep = min(max_clusters_to_keep, len(cluster_n_triangles))
        keep_indices = sorted_indices[:n_to_keep]

        keep_mask = np.isin(np.arange(len(cluster_n_triangles)), keep_indices) & large_enough

        triangles_to_keep = keep_mask[triangle_clusters]
        triangles_to_remove = ~triangles_to_keep

        mesh_clean.remove_triangles_by_mask(triangles_to_remove)
        mesh_clean.remove_unreferenced_vertices()
        mesh_clean.remove_degenerate_triangles()

        if smooth_iterations > 0:
            mesh_clean = mesh_clean.filter_smooth_laplacian(
                number_of_iterations=smooth_iterations,
                lambda_filter=smooth_lambda
            )
            mesh_clean.compute_vertex_normals()


        return mesh_clean
        