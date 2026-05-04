from pathlib import Path
import shutil
import cv2
from PIL import Image
import numpy as np
import logging
import random
from typing import List
import random

log = logging.getLogger(__name__)

def normalize_angle(angle: float) -> float:
    while angle < 0:
        angle += 360
    while angle >= 360:
        angle -= 360
    return angle

def extract_yaw_pitch(filename: Path) -> tuple:
    name = filename.stem 
    parts = name.split("_")
    if len(name.split('_')) != 4:
        yaw, pitch = map(float, name.split('_')[-2:])
        zoom = 1.
    else:
        yaw, pitch, zoom = map(float, name.split('_')[-3:])

    return (yaw, pitch)

def move_existing_files(texture_dir: Path, old_texture_dir: Path = None, yaw: float = 0, pitch: float = 0, epsilon: float = 1):
    moved_files = 0
    target_yaw = normalize_angle(yaw)

    for file in texture_dir.glob("*.png"):
        file_yaw, file_pitch = extract_yaw_pitch(file)
        file_yaw   = normalize_angle(file_yaw)
        if file_yaw is None or file_pitch is None:
            continue  # Skip files that don't match the expected pattern
        
        # Compare yaw and pitch with tolerance
        if abs(file_yaw - yaw) < epsilon and abs(file_pitch - pitch) < epsilon:
            if old_texture_dir is None:
                try:
                    file.unlink()
                    log.info(f"Deleted existing file {file.name}")
                    moved_files += 1
                except Exception as e:
                    log.error(f"Failed to delete file {file.name}: {e}")
            else:
                dest_path = old_texture_dir / file.name
                try:
                    shutil.move(str(file), str(dest_path))
                    log.info(f"Moved existing file {file.name} to {dest_path}")
                    moved_files += 1
                except Exception as e:
                    log.error(f"Failed to move file {file.name}: {e}")
    
    if moved_files == 0:
        log.info(f"No existing files matched yaw={yaw} and pitch={pitch} within epsilon={epsilon}")


def dilate_mask(
    mask: np.ndarray,
    fg_mask: np.ndarray,
    closing_kernel_size: tuple = (5, 5),
    dilate_kernel_size: tuple = (5, 5),
) -> Image.Image:
    """
    Applies dilation and morphological operations to a grayscale mask with respect to a foreground mask.

    Parameters:
    - gray_image: np.ndarray
        Grayscale image to process (2D array).
    - fg_mask: np.ndarray
        Foreground mask to constrain the dilation (2D binary array).
    - threshold_value: int, optional
        Threshold to binarize the grayscale image. Defaults to 200.
    - closing_kernel_size: tuple, optional
        Kernel size for the morphological closing operation. Defaults to (9, 9).
    - dilate_kernel_size: tuple, optional
        Kernel size for the dilation operation. Defaults to (4, 4).
    - gaussian_blur_sigma: float, optional
        Sigma value for Gaussian blurring. Defaults to 5.
    - final_threshold: int, optional
        Threshold to finalize the mask after blurring. Defaults to 50.

    Returns:
    - Image.Image
        The processed mask as a PIL Image.
    """
    fg_mask_binary = (fg_mask > 0).astype(np.uint8) * 255
    closing_kernel = np.ones(closing_kernel_size, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel, iterations=1)
    dilate_kernel = np.ones(dilate_kernel_size, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, dilate_kernel, iterations=1)
    combined_mask = cv2.bitwise_and(mask, fg_mask_binary)
    mask_pil = Image.fromarray(combined_mask)

    return mask_pil

def copy_texture_files(src_texture_dir: Path, dest_texture_dir: Path, num_samples: int):
    """
    Copies all .png files from the source texture directory to the destination texture directory.
    """
    # Gather all .png files matching the pattern "train_*.png"
    png_files: List[Path] = list(src_texture_dir.glob("train_*.png"))

    total_files = len(png_files)
    sampled_files = random.sample(png_files, total_files - num_samples)

    # Move each sampled file to the destination directory
    moved_files = 0
    for png_file in sampled_files:
        try:
            dest_path = dest_texture_dir / png_file.name
            shutil.copy(str(png_file), str(dest_path))
            #print(f"Moved '{png_file.name}' to '{dest_texture_dir}'.")
            moved_files += 1
        except Exception as e:
            log.info(f"Failed to move '{png_file.name}': {e}")

    log.info(f'Moved total of {moved_files} files')
