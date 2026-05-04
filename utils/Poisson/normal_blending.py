from pathlib import Path

import numpy as np
from PIL import Image

def normalize_for_visualization(normal_map):
    """
    Normalize the normal map for visualization:
    Scales the [-1, 1] range to [0, 255] for RGB image format.
    """
    return ((normal_map + 1) * 0.5 * 255).astype(np.uint8)

def load_and_normalize_normal_map(input_data):
    """
    Load a normal map from a PNG file path or PIL Image and normalize it to the range [-1, 1].
    Also return the alpha channel as the mask.
    """
    if isinstance(input_data, (str, Path)):
        img = Image.open(input_data).convert('RGBA')
    elif isinstance(input_data, Image.Image):
        img = input_data.convert('RGBA')
    elif isinstance(input_data, np.ndarray):
        img = Image.fromarray(input_data).convert('RGBA')
    else:
        raise ValueError("Input must be a file path, PIL Image, or numpy array.")

    img = np.array(img).astype(np.float32)
    
    if img.shape[-1] != 4:
        # If RGB, assume full mask
        if img.shape[-1] == 3:
            normals = img
            mask = np.ones(img.shape[:2], dtype=np.float32)
        else:
            raise ValueError(f"Image does not have 3 or 4 channels.")
    else:
        # Separate the RGB channels and the alpha channel (mask)
        normals = img[..., :3]
        mask = img[..., 3] / 255.0  # Normalize mask to [0, 1]
    
    # Assuming the normal map is stored in the range [0, 255]
    normals = normals / 255.0 * 2.0 - 1.0
    
    # Check for invalid normals (e.g., zero vectors)
    norm = np.linalg.norm(normals, axis=-1)
    # Avoid division by zero warnings if any
    # if np.any(norm == 0):
    #     raise ValueError(f"Invalid normal vector found in image.")
    
    return normals, mask

def blend_rnm(n1, n2):
    orientation = np.array([1, 1, 1])

    n1 = n1 * orientation
    n2 = n2 * orientation

    t = n1 + np.array([0, 0, 1])
    u = n2 * np.array([-1, -1, 1])
    
    dot_product = np.einsum('ijk,ijk->ij', t, u)

    r = t * dot_product[..., None] / t[..., -1][..., None] - u
    
    norm = np.linalg.norm(r, axis=2, keepdims=True)
    r = r / norm
    r = r * orientation

    return r

# def blend_rnm(n1, n2):
#     t = n1 + np.array([0, 0, 1])
#     u = n2 * np.array([1, -1, 1])
#     
#     dot_product = np.einsum('ijk,ijk->ij', t, u)
# 
#     r = t * dot_product[..., None] / t[..., -1][..., None] - u
#     
#     norm = np.linalg.norm(r, axis=2, keepdims=True)
#     r = r / norm
# 
#     return r

def blend_udn(n1, n2):
    """
    Blend two normal maps using the specified UDN blending function.
    n1, n2 are numpy arrays with shape (height, width, 3) representing the normal maps.
    """
    
    r = np.zeros_like(n1)
    r[..., :2] = n1[..., :2] + n2[..., :2]
    r[..., -1] = n1[..., -1]

    # Normalize the result
    norm = np.linalg.norm(r, axis=2, keepdims=True)
    normalized_r = r / norm
    
    return normalized_r

def blend_wo(n1, n2):
    """
    Blend two normal maps using the specified UDN blending function.
    n1, n2 are numpy arrays with shape (height, width, 3) representing the normal maps.
    """
    
    r = np.zeros_like(n1)
    r[..., :2] = n1[..., :2] + n2[..., :2]
    r[..., -1] = n1[..., -1] * n2[..., -1]

    # Normalize the result
    norm = np.linalg.norm(r, axis=2, keepdims=True)
    normalized_r = r / norm
    
    return normalized_r

def load_and_blend_maps(base_map_input, detail_map_input, l0_smoother=None):
    # Load the normal maps
    base_map, base_map_mask = load_and_normalize_normal_map(base_map_input)
    if l0_smoother is not None:
        base_map = l0_smoother.run(base_map)
        # Image.fromarray(normalize_for_visualization(base_map)).save(Path(base_map_path).parent / 'smoothed_base_normal.png')

    detail_map, detail_map_mask = load_and_normalize_normal_map(detail_map_input)

    base_map   *= base_map_mask[..., None]
    detail_map *= detail_map_mask[..., None]

    blended_map = blend_udn(base_map, detail_map)

    return Image.fromarray(normalize_for_visualization(blended_map))
