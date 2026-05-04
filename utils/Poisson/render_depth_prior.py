import os
from pathlib import Path

import torch
import numpy as np
import nvdiffrast.torch as dr

from torchvision.utils import save_image
from .core.remesh import calc_vertex_normals
from .util.func import make_yaw_pitch_ortho_cameras, ortho_proj
from .util.render import render
from .mesh.mesh import Mesh


def render_depth_prior(
        obj_path,
        im_res,
        pitch,
        yaw,
        zoom=1.
        ):

    start_mesh = Mesh().load(obj_path, resize=False)
    vertices, faces = start_mesh.v, start_mesh.f.to(torch.int64)
    
    pitches = torch.Tensor([torch.deg2rad(torch.tensor(pitch + 0.0))])
    yaws    = torch.Tensor([torch.deg2rad(torch.tensor(yaw   + 180.0))])

    mvs   = make_yaw_pitch_ortho_cameras(pitches, yaws)
    proj  = ortho_proj(device='cuda', max_x=0.5 * zoom, min_x=-0.5 * zoom, max_y=0.5 * zoom, min_y=-0.5 * zoom)
    glctx = dr.RasterizeGLContext()
    
    normals = calc_vertex_normals(vertices,faces)
    renders, depth = render(mvs, proj, glctx, [im_res, im_res], vertices, normals, faces)
    
    np_depth = depth.squeeze().cpu().detach().numpy()
    np_mvs   = mvs.squeeze().cpu().detach().numpy()
    np_alpha = renders[..., -1].squeeze().cpu().detach().numpy()

    return np_depth, np_alpha, np_mvs
