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

from torchvision.utils import make_grid

import nvdiffrast.torch as dr
from .mesh import Mesh, safe_normalize
from .cam_utils import orbit_camera, ortho_proj
from .renderer import render
from utils.crop import scale_foreground_object, split_rgba_to_rgb_a

def image_grid(imgs, cols):
    if len(imgs) == 0:
        raise ValueError("No images to display.")

    w, h = imgs[0].size
    rows = (len(imgs) + cols - 1) // cols  # Calculate the necessary number of rows

    grid = Image.new('RGBA', size=(cols * w, rows * h), color=0)  # Create a blank grid
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    
    return grid

def delete_class_attributes(obj):
    for attribute in dir(obj):
        # Make sure the attribute is not a built-in attribute or method
        if not attribute.startswith('__'):
            # delattr() function deletes an attribute if the object allows it
            try:
                delattr(obj, attribute)
            except AttributeError:
                log.warning(f"Cannot delete attribute {attribute}")


@torch.no_grad()
def remove_visible_faces(obj_path, partial_meshes_path, seen_partial_meshes_path, camera_plan, device='cuda'):
    mesh = Mesh().load(obj_path, resize=False)
    no_dup_mesh = remove_shared_verts(mesh)
    all_visible_faces = set()
    
    verts = no_dup_mesh.v
    faces = no_dup_mesh.f

    for idx, target_view in enumerate(camera_plan):
        t_yaw, t_pitch = target_view
        t_yaw, t_pitch = int(t_yaw), int(t_pitch)

        pose = orbit_camera(180 + t_pitch, t_yaw, opengl=True)
        proj = ortho_proj()
        out = render(mesh, pose, proj, h=1024 * 8, w=1024 * 8)

        visible_faces = out['t_id'].int().unique(sorted=False).tolist()
        all_visible_faces.update(visible_faces)
        visible_verts_idx  = faces[list(all_visible_faces)]       # (num_visible_faces,  3)
        #unique_vv_idx     = torch.unique(visible_verts_idx)   # (num_visible_verts, )

        vert_mask = torch.ones(len(verts), dtype=torch.bool)
        vert_mask[visible_verts_idx] = False
        #vert_mask[unique_vv_idx] = False
        invisible_verts   = verts[vert_mask]
        invisible_face_vi = torch.arange(len(invisible_verts), device=device).view(-1, 3)

        partial_obj_fp = os.path.join(partial_meshes_path, f'{t_yaw}_{t_pitch}.obj')
        Mesh.write_obj_with_only_verts_and_faces(partial_obj_fp, invisible_verts, invisible_face_vi)

        if idx == 0:
            vert_mask = torch.zeros(len(verts), dtype=torch.bool)
            vert_mask[visible_verts_idx] = True
            visible_verts   = verts[vert_mask]
            visible_face_vi = torch.arange(len(visible_verts), device=device).view(-1, 3)
            partial_obj_fp = os.path.join(seen_partial_meshes_path, f'{t_yaw}_{t_pitch}.obj')
            Mesh.write_obj_with_only_verts_and_faces(partial_obj_fp, visible_verts, visible_face_vi)

        del out

    del mesh
    del no_dup_mesh
    del verts
    del faces
    del all_visible_faces

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

@torch.no_grad()
def remove_shared_verts(mesh):
    vertices  = mesh.v.detach()
    vert_normals = mesh.vn.detach()
    triangles = mesh.f.detach()

    new_vertices  = torch.zeros((len(triangles) * 3, 3), dtype=torch.float32, device=mesh.device)
    new_triangles = torch.arange(len(triangles) * 3, device=mesh.device).reshape(-1, 3)
    new_vertices  = vertices[triangles].reshape(-1, 3)

    mesh.v  = new_vertices.float()  # torch.from_numpy(vertices).float().to(mesh.device)
    mesh.f  = new_triangles.int() # torch.from_numpy(triangles).int().to(mesh.device)
    mesh.vc = torch.zeros_like(mesh.v)

    mesh.auto_normal()

    return mesh

@torch.no_grad()
def segment_mesh(obj_path, 
                 seen_views_folder_path, 
                 seen_views, 
                 target_view=None, cos_threshold=0.1, skip_clean=False, device='cuda'):
    tic = time.perf_counter()
    mesh = Mesh().load(obj_path, resize=False)
    no_dup_mesh = remove_shared_verts(mesh)

    toc = time.perf_counter()

    verts = no_dup_mesh.v
    faces = no_dup_mesh.f
    face_normals = no_dup_mesh.face_normals
    faces_sim_cache = -1. * torch.ones(len(faces)).to(device) # cosine similarity cutoff
    
    uv_verts  = torch.zeros(len(verts), 2).to(device)
    uv_faces  = faces

    seen_rgbs = []
    seen_rgbs_scale = []
    #seen_seams = []
    cos_sim_maps = []
    all_unique_vv_idx = []
    all_unique_vf_idx = []

    seen_rgbs_pil = []
    cos_sim_maps_pil = []

    # 1. Get visible portion of the mesh 2. Identify faces to update
    for idx, view in enumerate(seen_views):
        yaw, ele = view
        seen_rgb  = Image.open(os.path.join(seen_views_folder_path, f'rgb_{int(yaw)}_{int(ele)}.png'))#.convert('RGB').resize((1024, 1024))
        edge_mask = Image.open(os.path.join(seen_views_folder_path, f'edge_{int(yaw)}_{int(ele)}.png')).convert('L').resize((1024, 1024))
        # if idx != 0:
        #     seen_seam = Image.open(os.path.join(seen_views_folder_path, f'seam_{int(yaw)}_{int(ele)}.png'))
        # else:
        #     seen_seam = Image.new('RGB', (1024, 1024), (0, 0, 0))
        #     #seen_seam = Image.open(os.path.join(seen_views_folder_path, f'seam_{int(yaw)}_{int(ele)}.png'))

        #seen_rgb = ImageOps.mirror(seen_rgb)
        
        seen_rgb_ori = np.array(seen_rgb) / 255.
        seen_rgb_ori = torch.Tensor(seen_rgb_ori).to(device)

        _, fg_mask_pil = split_rgba_to_rgb_a(seen_rgb)
        seen_rgb, scale = scale_foreground_object(seen_rgb, fg_mask_pil)
        #scale = 1.

        seen_rgbs_pil.append(seen_rgb)
        seen_rgb_np = np.array(seen_rgb) / 255.
        seen_rgb = torch.Tensor(seen_rgb_np).to(device)
        seen_rgbs.append(seen_rgb)

        edge_mask_np = np.array(edge_mask) / 255.
        edge_mask_np = edge_mask_np * seen_rgb_np[..., -1]
        edge_mask = torch.Tensor(edge_mask_np).to(device)

        # seen_seam = np.array(seen_seam) / 255.
        # seen_seam = torch.Tensor(seen_seam).to(device)
        # seen_seams.append(seen_seam)

        res = 1024 * 6

        pose = orbit_camera(180 + ele, yaw, opengl=True)
        proj = ortho_proj()
        out  = render(no_dup_mesh, pose, proj, h=res, w=res)

        seen_rgb = torch.Tensor(seen_rgb).to(device)
        cos_sim_map = out['viewcos'].squeeze().cpu().numpy()
        cos_sim_map = cv2.resize(cos_sim_map, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        cos_sim_map = repeat(cos_sim_map, 'h w -> h w c', c=3)
        cos_sim_map_pil = Image.fromarray((cos_sim_map * 255.).astype(np.uint8))
        # !!!
        cos_sim_map_pil, _ = scale_foreground_object(cos_sim_map_pil, fg_mask_pil)
        cos_sim_maps_pil.append(cos_sim_map_pil)

        cos_sim_maps.append(torch.Tensor(cos_sim_map).to(device))
        
        seen_rgb_mask = rearrange(seen_rgb_ori[..., -1], 'h w -> 1 1 h w')
        edge_mask     = rearrange(edge_mask, 'h w -> 1 1 h w')
        texture_mask  = F.interpolate(seen_rgb_mask, size=res, mode='nearest-exact').squeeze() == 0
        edge_mask     = F.interpolate(edge_mask, size=res, mode='nearest-exact').squeeze() == 1

        all_rendered_faces  = out['t_id'].int().unique(sorted=False)
        # Even if the face is rendered, we invalidate it if the face has even a single invisible pixel
        # WARN: So don't change the ~toch.isin! the negation is there for a purpose
        invisible_faces     = out['t_id'][texture_mask.unsqueeze(0)].int().unique(sorted=False)
        edge_faces          = out['t_id'][edge_mask.unsqueeze(0)].int().unique(sorted=False)
        face_mask           = ~torch.isin(all_rendered_faces, invisible_faces)
        visible_faces       = all_rendered_faces[face_mask]
        #edge_faces          = all_rendered_faces[torch.isin(all_rendered_faces, edge_faces)]
        edge_faces          = visible_faces[torch.isin(visible_faces, edge_faces)]

        visible_verts_idx   = faces[visible_faces]       # (num_visible_faces,  3)
        unique_vv_idx       = torch.unique(visible_verts_idx)   # (num_visible_verts, )

        all_faces_to_update = []
        faces_sim_cache_vf  = faces_sim_cache[visible_faces]
        
        pose = torch.from_numpy(pose.astype(np.float32)).to(device)
        proj = torch.from_numpy(proj.astype(np.float32)).to(device)
        all_faces_cos_sim   = (face_normals @ pose[:3, :3])[:, 2]
        all_faces_cos_sim[edge_faces] = torch.where(all_faces_cos_sim[edge_faces] > 0.4, 0., -1.) # To avoid ledge fall-offs
        vis_faces_cos_sim   = all_faces_cos_sim[visible_faces]

        # 1. Identify where updates are needed
        update_mask = vis_faces_cos_sim > faces_sim_cache_vf + 0.4
        # 2. Update faces_sim_cache_vf where the condition is True
        faces_sim_cache_vf[update_mask] = vis_faces_cos_sim[update_mask]
        # 3. Use the mask to select from visible_verts_idx and update all_faces_to_update
        all_faces_to_update = visible_verts_idx[update_mask]

        faces_sim_cache[visible_faces] = faces_sim_cache_vf 
        f_w_vv = torch.vstack([all_faces_to_update])
        unique_vv_idx = torch.unique(f_w_vv)   # (num_visible_verts, )
        all_unique_vv_idx.append(unique_vv_idx)
        all_unique_vf_idx.append(visible_faces)

        # Project the world vertices to screen space to sample texture values
        visible_verts = verts[unique_vv_idx, :]

        vv_cam  = torch.matmul(F.pad(visible_verts, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float()
        vv_clip = vv_cam @ proj.T
        
        uv_verts[unique_vv_idx, :] = (vv_clip[:, :2] * scale + 1.) * 0.5
        uv_verts[unique_vv_idx, 0] /= 4 #len(seen_views) 
        uv_verts[unique_vv_idx, 0] += (idx %  4) / 4 #idx * (1 / len(seen_views))
        n_rows = math.ceil(len(seen_views) / 4.)
        uv_verts[unique_vv_idx, 1] /= n_rows #idx * (1 / len(seen_views))
        uv_verts[unique_vv_idx, 1] += (idx // 4) / n_rows #idx * (1 / len(seen_views))

    textured_mesh   = copy(no_dup_mesh)

    texture_map = image_grid(seen_rgbs_pil, cols=4)
    texture_map = np.array(texture_map) / 255.
    texture_map = torch.Tensor(texture_map).to(device)

    textured_mesh.vt      = uv_verts
    textured_mesh.ft      = uv_faces
    textured_mesh.albedo  = texture_map
    #textured_mesh.sim_map = torch.hstack(cos_sim_maps)
    cos_sim_map = image_grid(cos_sim_maps_pil, cols=4)
    cos_sim_map = np.array(cos_sim_map) / 255.
    cos_sim_map = torch.Tensor(cos_sim_map).to(device)
    textured_mesh.sim_map = cos_sim_map

    # NOTE: We don't want to preserve mesh region with cos_sim less than 0.4

    if target_view is not None:

        t_yaw, t_pitch = target_view
        t_yaw, t_pitch = int(t_yaw), int(t_pitch)

        pose = orbit_camera(180 + t_pitch, t_yaw, opengl=True)
        proj = ortho_proj()
        out  = render(textured_mesh, pose, proj, h=1024 * 6, w=1024 * 6)

        all_faces = out['t_id'].int().unique(sorted=False)
        view_cos  = out['viewcos_cache'].squeeze()[..., 0]
        cos_mask  = view_cos < 0.4
        invisible_faces     = out['t_id'][cos_mask.unsqueeze(0)].int().unique(sorted=False)
        face_mask           = torch.isin(all_faces, invisible_faces)
        visible_faces       = all_faces[face_mask]
        visible_verts_idx   = faces[visible_faces]       # (num_visible_faces,  3)
        unique_vv_idx       = torch.unique(visible_verts_idx)   # (num_visible_verts, )
        
        vert_mask = torch.zeros(len(verts), dtype=torch.bool)
        vert_mask[torch.hstack(all_unique_vv_idx)] = True
        #vert_mask[unique_vv_idx] = False # cos_thresholding

        visible_verts   = verts[vert_mask]
        visible_face_vi = torch.arange(len(visible_verts), device=device).view(-1, 3)
        seen_partial = (visible_verts, visible_face_vi)
    else:
        seen_partial = None


    del mesh
    del no_dup_mesh
    del all_unique_vv_idx 
    del all_unique_vf_idx 

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return textured_mesh, seen_partial

