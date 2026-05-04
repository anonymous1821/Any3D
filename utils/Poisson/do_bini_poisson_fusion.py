import os
import time
import subprocess
from pathlib import Path

import cv2
import trimesh
import pymeshlab
import torch
import open3d as o3d

from utils.Poisson import (
    render_depth_prior, 
    run_bilateral_integration, 
    load_normal_map_with_alpha,
)
from utils.Poisson.mesh import Mesh
from utils.Projection.scripts.remove_seen_faces import remove_seen_faces
from utils.etc import normalize_angle

def convert_obj_to_ply(obj_path):
    """
    Converts an OBJ file to a PLY file using trimesh.
    
    Parameters:
    - obj_path (str or Path): Path to the input OBJ file.
    - ply_path (str or Path, optional): Path to save the output PLY file. Same name but with a .ply extension.

    """
    obj_path = Path(obj_path)
    ply_path = obj_path.with_suffix('.ply')
    
    # Load the OBJ file into a Trimesh object
    mesh = trimesh.load(obj_path)
    # mesh.vertex_normals

    # # Export the mesh to PLY format
    # mesh.export(ply_path)

    result = trimesh.exchange.ply.export_ply(mesh, vertex_normal=True)
    output_file = open(ply_path, "wb+")
    output_file.write(result)
    output_file.close()

    return ply_path

def convert_ply_to_obj(ply_path):
    ply_path = Path(ply_path)
    obj_path = ply_path.with_suffix('.obj')
    
    # Load the OBJ file into a Trimesh object
    mesh = trimesh.load(ply_path)

    # Export the mesh to OBJ format
    mesh.vertex_normals
    mesh.export(obj_path)

    return obj_path

def process_poisson(
    angle,
    current_obj_path, 
    bini_meshes,
    texture_dir, 
    mask_path, 
    normal_path, 
    poisson_save_dir, 
    partial_meshes_dir, 
    #bini_save_dir, 
    seen_yaw_pitch_list,
    poisson_bin_fp,
    im_res=1024, 
    poisson_depth=9,
    envelope_depth=None,
    seen_thresh=0.3,
    cleanup=False,
):
    """
    Process Poisson reconstruction using specified directories and parameters.

    Args:
        coarse_fp (str): Filepath to the coarse mesh.
        texture_dir (str): Directory containing texture files.
        mask_dir (str): Directory containing masks.
        normal_dir (str): Directory containing normal maps.
        poisson_save_dir (str): Directory to save Poisson surfaces.
        partial_meshes_dir (str): Directory to save partial meshes.
        bini_save_dir (str): Directory to save bilateral integration surfaces.
        im_res (int): Image resolution. Default is 1024.
        poisson_depth (int): Depth for Poisson reconstruction. Default is 9.
        cleanup (bool): Whether to remove intermediate files. Default is False.

    Returns:
        None
    """
    poisson_start = time.perf_counter()
    Path(poisson_save_dir).mkdir(parents=True, exist_ok=True)
    Path(partial_meshes_dir).mkdir(parents=True, exist_ok=True)

    obj_fp = current_obj_path
    yaw    = angle.yaw
    pitch  = angle.pitch

    # NOTE: Remove Seen view and processing view Surfaces from Prev Mesh
    vis_mesh_fp   = remove_seen_faces(obj_fp, partial_meshes_dir, texture_dir, seen_yaw_pitch_list + [angle], angle, dot_threshold=seen_thresh)
    # vis_mesh_fp   = remove_seen_faces(obj_fp, partial_meshes_dir, texture_dir, [angle], angle, dot_threshold=seen_thresh)
    seen_face_removal_end = time.perf_counter()
    vis_mesh      = Mesh().load(vis_mesh_fp, resize=False, device='cpu')
    
    if cleanup:
        try:
            os.remove(vis_mesh_fp)
        except OSError:
            pass

    merge_mesh    = Mesh.merge(bini_meshes + [vis_mesh])

    # Pre clean
    ms = pymeshlab.MeshSet()
    pm_mesh = pymeshlab.Mesh(vertex_matrix=merge_mesh.v.numpy(), 
                             face_matrix=merge_mesh.f.numpy())
    ms.add_mesh(pm_mesh, "merge_mesh")

    ms.meshing_merge_close_vertices()
    ms.compute_selection_by_self_intersections_per_face()
    ms.meshing_remove_selected_faces()

    # Remove small islands
    ms.meshing_remove_connected_component_by_diameter(
        mincomponentdiag=pymeshlab.PercentageValue(1.),
        removeunref=True
    )
    
    # 5. Retrieve final mesh data
    final_mesh     = ms.current_mesh()
    final_vertices = final_mesh.vertex_matrix()
    final_faces    = final_mesh.face_matrix()

    # merge_mesh = Mesh(
    #     device='cpu',
    #     v=torch.Tensor(final_vertices).float(),
    #     f=torch.Tensor(final_faces).int(),
    #     )

    o3d_mesh           = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices  = o3d.utility.Vector3dVector(final_vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(final_faces)
    o3d_mesh.compute_vertex_normals()

    point_sample_start = time.perf_counter()
    sampled_pcd_poisson = o3d_mesh.sample_points_uniformly(number_of_points=1_000_000)
    #sampled_pcd_poisson = o3d_mesh.sample_points_poisson_disk(number_of_points=1_000_000) #1_000_000
    point_sample_end = time.perf_counter()

    merge_mesh_fp = os.path.join(partial_meshes_dir, f"merged_{yaw:.1f}_{pitch:.1f}.ply")
    #merge_mesh.write(merge_mesh_fp)
    o3d.io.write_point_cloud(merge_mesh_fp, sampled_pcd_poisson)
    poisson_fp    = os.path.join(poisson_save_dir, f"poisson_{yaw:.1f}_{pitch:.1f}.ply")

    poisson_start = time.perf_counter()
    # NOTE: Run envelope Poisson
    command = [
        f"{poisson_bin_fp}",
        "--in", f"{merge_mesh_fp}",
        "--envelope", f"{obj_fp}",
        "--out", f"{poisson_fp}",
        "--depth", f"{poisson_depth}",
        "--density",
        "--performance"
    ]

    if envelope_depth is not None:
        command.extend(["--envelopeDepth", f"{envelope_depth}"])
    
    result = subprocess.run(command, capture_output=True, text=True)
    poisson_end  = time.perf_counter()
    
    if cleanup:
        try:
            os.remove(merge_mesh_fp)
        except OSError:
            pass
        try:
            os.rmdir(partial_meshes_dir)
        except OSError:
            pass
            
    return poisson_fp
