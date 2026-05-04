import math
import mathutils
import json
import os
import numpy as np 
from typing import Dict, Optional

def transforms_from_angles(
    yaws,
    pitches,
    fovs,
    radius,
    file_paths=None,
    center=(0.0, 0.0, 0.0),
    up=(0.0, 0.0, 1.0),
):
    """
    Convert yaw / pitch / fov camera parameters into Blender camera matrices.

    yaw   : rotation around Z axis
    pitch : elevation angle
    radius: camera distance from center
    """

    def to_list(x):
        return x if isinstance(x, (list, tuple)) else [x]

    yaws_l = to_list(yaws)
    pitches_l = to_list(pitches)
    fovs_l = to_list(fovs)

    n = max(len(yaws_l), len(pitches_l), len(fovs_l))

    def broadcast(lst):
        if len(lst) == 1:
            return lst * n
        if len(lst) == n:
            return lst
        return lst + [lst[-1]] * (n - len(lst))

    yaws_l = broadcast(yaws_l)
    pitches_l = broadcast(pitches_l)
    fovs_l = broadcast(fovs_l)

    if file_paths is None:
        file_paths = [None] * n
    elif isinstance(file_paths, str):
        file_paths = [file_paths] * n
    elif len(file_paths) == 1:
        file_paths = file_paths * n

    frames = []

    center_vec = mathutils.Vector(center)
    up_vec = mathutils.Vector(up)

    for yaw, pitch, fov, fname in zip(yaws_l, pitches_l, fovs_l, file_paths):

        yaw = float(yaw)
        pitch = float(pitch)
        fov = float(fov)

        if abs(yaw) > 2 * math.pi:
            yaw = math.radians(yaw)

        if abs(pitch) > 2 * math.pi:
            pitch = math.radians(pitch)

        if fov > 2 * math.pi:
            fov = math.radians(fov)

        x = radius * math.cos(pitch) * math.sin(yaw)
        y = -radius * math.cos(pitch) * math.cos(yaw)
        z = radius * math.sin(pitch)

        loc = mathutils.Vector((x, y, z))

        forward = (center_vec - loc).normalized()

        if abs(forward.dot(up_vec)) > 0.999:
            up_vec = mathutils.Vector((0.0, 1.0, 0.0))

        right = up_vec.cross(forward).normalized()
        true_up = forward.cross(right).normalized()

        back = -forward

        mat = mathutils.Matrix(
            (
                (right.x, true_up.x, back.x, loc.x),
                (right.y, true_up.y, back.y, loc.y),
                (right.z, true_up.z, back.z, loc.z),
                (0.0, 0.0, 0.0, 1.0),
            )
        )

        mat_list = [[float(v) for v in row] for row in mat]

        frame = {
            "transform_matrix": mat_list,
            "camera_angle_x": fov,
        }

        frames.append(frame)

    return frames


def pixel2point(u: int, v: int, depth: np.ndarray, transform: Dict) -> np.ndarray:
    height, width = depth.shape
    
    fov_x = transform['camera_angle_x']
    focal_length = 0.5 * width / np.tan(0.5 * fov_x)
    
    cx = width / 2.0
    cy = height / 2.0
    
    z_val = depth[v, u]
    
    v= height - v - 1
    u = width - u - 1

    x_cam = (u - cx) * z_val / focal_length
    y_cam = (v - cy) * z_val / focal_length
    z_cam = z_val
    
    p_cam_blender = np.array([x_cam, -y_cam, -z_cam, 1.0])
    
    c2w = np.array(transform['transform_matrix'])
    p_world = c2w @ p_cam_blender
    
    return p_world[:3]


if __name__ == "__main__":
    
    yaws = [0, 45, 135, 225, 315]
    pitch = 30
    fov = 49
    radius = 1.6

    # yaws = [0, 90, 180, 270]
    # pitch = 10 

    # transforms = transforms_from_angles(
    #     yaws, 
    #     pitch,
    #     fov, 
    #     radius
    # )
    # with open(f"./transforms4.json", "w") as f:
    #     json.dump(transforms, f, indent=4)
    for i, yaw in enumerate(yaws):
        transforms = transforms_from_angles(
            yaw,
            pitch,
            fov,
            radius,
            file_paths=f"{i:03d}.png",
        )
        with open(f"./transforms/{i:03d}.json", "w") as f:
            json.dump(transforms, f, indent=4)


