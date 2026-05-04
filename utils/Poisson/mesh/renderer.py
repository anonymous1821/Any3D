import os
import math
import time
from copy import copy
from datetime import datetime

import cv2
import trimesh
import open3d as o3d
import numpy as np
from PIL import Image, ImageOps
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

import nvdiffrast.torch as dr
from .mesh import Mesh, safe_normalize
from .cam_utils import orbit_camera, ortho_proj

def render(mesh, pose, proj, h, w, max_mip_level=10, device='cuda', glctx=dr.RasterizeGLContext()):
    results = {}
    # get v
    v = mesh.v

    # get v_clip and render rgb
    pose = torch.from_numpy(pose.astype(np.float32)).to(v.device)
    proj = torch.from_numpy(proj.astype(np.float32)).to(v.device)
    v_cam = torch.matmul(F.pad(v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
    v_clip = v_cam @ proj.T
    rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (h, w))
    t_id = rast[..., 3] - 1
    
    alpha_raw = (rast[..., 3:] > 0).float()
    alpha = dr.antialias(alpha_raw, rast, v_clip, mesh.f).squeeze(0).clamp(0, 1) # [H, W, 3]

    v_cam_z = v_cam[..., [2]]
    depth, _ = dr.interpolate(v_cam_z, rast, mesh.f) # [1, H, W, 1]
    depth = depth.squeeze()

    depth_min   = torch.max(depth[depth != 0.])
    depth_max   = torch.min(depth[depth != 0.])
    depth_bg_depth = depth_max * 1.05
    depth[depth == 0.] = depth_bg_depth

    ori_depth_no_scale = depth
    ori_depth = (depth - depth_min) / (depth_bg_depth - depth_min)
    depth = ((depth - depth_min) / (depth_bg_depth - depth_min) - 0.5) * 2.

    
    # rgb texture (UV texture)
    if mesh.vt is not None and mesh.albedo is not None:
        texc, texc_db = dr.interpolate(mesh.vt.unsqueeze(0).contiguous(), rast, mesh.ft, rast_db=rast_db, diff_attrs='all')
        albedo_no_aa = dr.texture(mesh.albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level) # [1, H, W, 3]
        albedo = dr.antialias(albedo_no_aa, rast, v_clip, mesh.f).squeeze(0).clamp(0, 1) # [H, W, 3]

    elif mesh.vc is not None: # if vertex color
        albedo_no_aa, _ = dr.interpolate(mesh.vc.unsqueeze(0).contiguous(), rast, mesh.f)
        albedo = dr.antialias(albedo_no_aa, rast, v_clip, mesh.f).squeeze(0).clamp(0, 1) # [H, W, 3]
    else:
        albedo_no_aa = torch.ones_like(alpha)
        albedo = torch.ones_like(alpha)
    
    # get vn and render normal
    vn = mesh.vn
    
    normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, mesh.fn)
    normal = safe_normalize(normal[0])

    # rotated normal (where [0, 0, 1] always faces camera)
    rot_normal = normal @ pose[:3, :3]

    # rot normal z axis is exactly viewdir-normal cosine
    viewcos = rot_normal[..., [2]].abs() # double-sided

    # replace background
    if albedo.shape[-1] == 4:
        bg = torch.tensor([.5, .5, .5, 1], dtype=torch.float32, device=device)
        # bg = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=device)
    else:
        bg = torch.ones(albedo.shape[-1], dtype=torch.float32, device=device)
    bg_normal = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)

    albedo = alpha * albedo + (1 - alpha) * bg
    hard_albedo = alpha_raw * albedo_no_aa + (1 - alpha_raw) * bg
    normal = alpha * normal + (1 - alpha) * bg_normal
    rot_normal = alpha * rot_normal + (1 - alpha) * bg_normal

    # extra texture (hard coded)
    if hasattr(mesh, 'cnt'):
        cnt = dr.texture(mesh.cnt.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')
        cnt = dr.antialias(cnt, rast, v_clip, mesh.f).squeeze(0) # [H, W, 3]
        cnt = alpha * cnt + (1 - alpha) * 1 # 1 means no-inpaint in background
        results['cnt'] = cnt
    
    if hasattr(mesh, 'viewcos_cache'):
        viewcos_cache = dr.texture(mesh.viewcos_cache.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')
        viewcos_cache = dr.antialias(viewcos_cache, rast, v_clip, mesh.f).squeeze(0) # [H, W, 3]
        results['viewcos_cache'] = viewcos_cache

    if hasattr(mesh, 'ori_albedo'):
        ori_albedo = dr.texture(mesh.ori_albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')
        ori_albedo = dr.antialias(ori_albedo, rast, v_clip, mesh.f).squeeze(0) # [H, W, 3]
        ori_albedo = alpha * ori_albedo + (1 - alpha) * bg
        results['ori_image'] = ori_albedo

    if hasattr(mesh, 'sim_map'):
        sim_cache = dr.texture(mesh.sim_map.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level) # [1, H, W, 3]
        sim_cache = dr.antialias(sim_cache, rast, v_clip, mesh.f).squeeze(0).clamp(0, 1) # [H, W, 3]
        results['viewcos_cache'] = sim_cache
    
    # all shaped as [H, W, C]
    results['image'] = albedo
    results['image_no_aa'] = hard_albedo
    results['alpha'] = alpha
    results['alpha_raw'] = alpha_raw
    results['ori_depth'] = ori_depth
    results['ori_depth_no_scale'] = ori_depth_no_scale
    results['depth'] = depth
    results['fg_depth_min'] = depth_min.item()
    results['fg_depth_max'] = depth_max.item()
    results['normal'] = normal # in [-1, 1]
    results['rot_normal'] = rot_normal # in [-1, 1]
    results['viewcos'] = viewcos
    results['t_id'] = t_id

    if mesh.vt is not None and mesh.albedo is not None:
        results['uvs'] = texc.squeeze(0)

    return results
