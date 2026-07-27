"""
Canonical orientation helpers for TRELLIS-style Gaussian PLY / GLB assets.

Geometry conventions follow QwenSDS gen_init_models/caption_view.py:
- File (Y-up) -> Render (Z-up): FILE_TO_RENDER
- PLY target front: yaw=0, pitch=0 (gaussian cam convention, yaw negated)
- GLB target front: yaw=180, pitch=0 (mesh cam convention)
"""
from __future__ import annotations

import glob
import os
import shutil
from typing import List, Optional, Sequence, Tuple

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R

from utils.logger import get_logger, setup_logging

setup_logging()
log = get_logger(__name__)

# File (Y-up GLB/PLY) -> Render (Z-up). Matches GaussianModel.load_ply default.
FILE_TO_RENDER = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

# 8 orthogonal views: 4 side + 4 overhead (same schedule as caption_view.py)
ORTHO_YAWS: List[float] = [0, 90, 180, -90, 0, 90, 180, -90]
ORTHO_PITCHES: List[float] = [0, 0, 0, 0, 90, 90, 90, 90]

PLY_TARGET_YAW = 0.0
PLY_TARGET_PITCH = 0.0
GLB_TARGET_YAW = 180.0
GLB_TARGET_PITCH = 0.0


def safe_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return v
    return v / norm


def get_cam_rot(yaw_deg: float, pitch_deg: float, gaussian: bool = True) -> np.ndarray:
    """Camera-to-world rotation matching utils3d / TRELLIS look-at (Z-up)."""
    # Negate yaw for Gaussian renderer Counter-Clockwise convention
    # (Input 90 is Left/-X, Input 270 is Right/+X)
    if gaussian:
        yaw_deg = -yaw_deg

    yaw_deg = yaw_deg % 360
    y_rad = np.deg2rad(yaw_deg)
    p_rad = np.deg2rad(pitch_deg)

    cam_pos = np.array([
        np.sin(y_rad) * np.cos(p_rad),
        np.cos(y_rad) * np.cos(p_rad),
        np.sin(p_rad),
    ])
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])

    forward = safe_normalize(target - cam_pos)
    right = safe_normalize(np.cross(forward, up))
    if np.linalg.norm(right) < 1e-6:
        # Pitch ~90: preserve yaw rotation
        right = np.array([-np.cos(y_rad), np.sin(y_rad), 0.0])

    new_up = safe_normalize(np.cross(right, forward))

    mat = np.eye(3)
    mat[:, 0] = right
    mat[:, 1] = new_up
    mat[:, 2] = forward  # TRELLIS uses Forward (not -Forward) as 3rd basis
    return mat


def _backup_once(path: str) -> None:
    backup_path = path + ".bak"
    if not os.path.exists(backup_path):
        shutil.copy(path, backup_path)


def rotate_ply(
    ply_path: str,
    yaw_deg: float,
    pitch_deg: float,
    backup: bool = True,
) -> None:
    """Rotate a Gaussian PLY so (yaw, pitch) becomes the canonical front (0, 0)."""
    plydata = PlyData.read(ply_path)

    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    r0 = np.asarray(plydata.elements[0]["rot_0"])
    r1 = np.asarray(plydata.elements[0]["rot_1"])
    r2 = np.asarray(plydata.elements[0]["rot_2"])
    r3 = np.asarray(plydata.elements[0]["rot_3"])
    # Scipy uses [x, y, z, w]
    rots = np.stack((r1, r2, r3, r0), axis=1)

    U = FILE_TO_RENDER
    R_sel = get_cam_rot(yaw_deg, pitch_deg, gaussian=True)
    R_target = get_cam_rot(PLY_TARGET_YAW, PLY_TARGET_PITCH, gaussian=True)
    R_obj = R_target @ R_sel.T
    R_ply_mat = U.T @ R_obj @ U

    rot_ply = R.from_matrix(R_ply_mat)
    xyz_new = rot_ply.apply(xyz)
    rots_new = (rot_ply * R.from_quat(rots)).as_quat()  # [x, y, z, w]

    # Rebuild vertex element (in-place slice writes can fail with plyfile on large PLYs)
    vertex = plydata.elements[0].data.copy()
    vertex["x"] = xyz_new[:, 0]
    vertex["y"] = xyz_new[:, 1]
    vertex["z"] = xyz_new[:, 2]
    vertex["rot_0"] = rots_new[:, 3]  # w
    vertex["rot_1"] = rots_new[:, 0]  # x
    vertex["rot_2"] = rots_new[:, 1]  # y
    vertex["rot_3"] = rots_new[:, 2]  # z
    new_elements = [PlyElement.describe(vertex, plydata.elements[0].name)]
    new_elements.extend(plydata.elements[1:])
    out = PlyData(new_elements, text=plydata.text)

    if backup:
        _backup_once(ply_path)
    out.write(ply_path)
    log.info(f"Rotated PLY {ply_path} (sel yaw={yaw_deg}, pitch={pitch_deg})")


def rotate_glb(
    glb_path: str,
    yaw_deg: float,
    pitch_deg: float,
    backup: bool = True,
) -> None:
    """Rotate a GLB so (yaw, pitch) becomes the mesh canonical front (180, 0)."""
    import trimesh

    mesh = trimesh.load(glb_path)
    U = FILE_TO_RENDER
    R_sel = get_cam_rot(yaw_deg, pitch_deg, gaussian=False)
    R_target = get_cam_rot(GLB_TARGET_YAW, GLB_TARGET_PITCH, gaussian=False)
    R_obj = R_target @ R_sel.T
    R_file_mat = U.T @ R_obj @ U

    transform = np.eye(4)
    transform[:3, :3] = R_file_mat
    mesh.apply_transform(transform)

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    if backup:
        _backup_once(glb_path)
    mesh.export(glb_path)
    log.info(f"Rotated GLB {glb_path} (sel yaw={yaw_deg}, pitch={pitch_deg})")


def find_ply_files(asset_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(asset_dir, "*.ply")))


def find_glb_files(asset_dir: str, prefer_mesh_glb: bool = True) -> List[str]:
    mesh_glb = os.path.join(asset_dir, "mesh.glb")
    if prefer_mesh_glb and os.path.exists(mesh_glb):
        return [mesh_glb]
    return sorted(glob.glob(os.path.join(asset_dir, "*.glb")))


def apply_canonical_rotation(
    asset_dir: Optional[str] = None,
    sel_yaw: float = 0.0,
    sel_pitch: float = 0.0,
    ply_paths: Optional[Sequence[str]] = None,
    glb_paths: Optional[Sequence[str]] = None,
    backup: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Apply PLY/GLB canonical rotations for a selected ortho view.

    Returns (rotated_ply_paths, rotated_glb_paths).
    """
    rotated_ply: List[str] = []
    rotated_glb: List[str] = []

    if ply_paths is None:
        ply_paths = find_ply_files(asset_dir) if asset_dir else []
    if glb_paths is None:
        glb_paths = find_glb_files(asset_dir) if asset_dir else []

    if abs(sel_yaw - PLY_TARGET_YAW) > 1e-5 or abs(sel_pitch - PLY_TARGET_PITCH) > 1e-5:
        for ply_path in ply_paths:
            rotate_ply(ply_path, sel_yaw, sel_pitch, backup=backup)
            rotated_ply.append(ply_path)
    else:
        log.info("PLY already at canonical front (yaw=0, pitch=0); skip rotate")

    if abs(sel_yaw - GLB_TARGET_YAW) > 1e-5 or abs(sel_pitch - GLB_TARGET_PITCH) > 1e-5:
        for glb_path in glb_paths:
            rotate_glb(glb_path, sel_yaw, sel_pitch, backup=backup)
            rotated_glb.append(glb_path)
    else:
        if glb_paths:
            log.info("GLB already at canonical front (yaw=180, pitch=0); skip rotate")

    return rotated_ply, rotated_glb


def ortho_index_to_yaw_pitch(idx_0_based: int) -> Tuple[float, float]:
    if idx_0_based < 0 or idx_0_based >= len(ORTHO_YAWS):
        raise IndexError(f"Ortho view index {idx_0_based} out of range [0, {len(ORTHO_YAWS)})")
    return float(ORTHO_YAWS[idx_0_based]), float(ORTHO_PITCHES[idx_0_based])
