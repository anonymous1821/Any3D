"""
Modified Bilateral Normal Integration (BiNI)
"""
__author__ = "Xu Cao <cao.xu@ist.osaka-u.ac.jp>; Yuliang Xiu <yuliang.xiu@tue.mpg.de>"
__copyright__ = "Copyright (C) 2022 Xu Cao; Yuliang Xiu"
__contributors__ = "Nuri Ryu <ryunuri@postech.ac.kr>"
__version__ = "2.1"

import pyvista as pv
import cupy as cp
import numpy as np
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import cg
from tqdm.auto import tqdm
import time

import cv2
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from PIL import Image

pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)

# Define helper functions for moving masks in different directions
def move_left(mask): return cp.pad(mask,((0,0),(0,1)),'constant',constant_values=0)[:,1:]  # Shift the input mask array to the left by 1, filling the right edge with zeros.
def move_right(mask): return cp.pad(mask,((0,0),(1,0)),'constant',constant_values=0)[:,:-1]  # Shift the input mask array to the right by 1, filling the left edge with zeros.
def move_top(mask): return cp.pad(mask,((0,1),(0,0)),'constant',constant_values=0)[1:,:]  # Shift the input mask array up by 1, filling the bottom edge with zeros.
def move_bottom(mask): return cp.pad(mask,((1,0),(0,0)),'constant',constant_values=0)[:-1,:]  # Shift the input mask array down by 1, filling the top edge with zeros.
def move_top_left(mask): return cp.pad(mask,((0,1),(0,1)),'constant',constant_values=0)[1:,1:]  # Shift the input mask array up and to the left by 1, filling the bottom and right edges with zeros.
def move_top_right(mask): return cp.pad(mask,((0,1),(1,0)),'constant',constant_values=0)[1:,:-1]  # Shift the input mask array up and to the right by 1, filling the bottom and left edges with zeros.
def move_bottom_left(mask): return cp.pad(mask,((1,0),(0,1)),'constant',constant_values=0)[:-1,1:]  # Shift the input mask array down and to the left by 1, filling the top and right edges with zeros.
def move_bottom_right(mask): return cp.pad(mask,((1,0),(1,0)),'constant',constant_values=0)[:-1,:-1]  # Shift the input mask array down and to the right by 1, filling the top and left edges with zeros.


def generate_dx_dy(mask, nz_horizontal, nz_vertical, step_size=1):
    # pixel coordinates
    # ^ vertical positive
    # |
    # |
    # |
    # o ---> horizontal positive
    num_pixel = cp.sum(mask)

    # Generate an integer index array with the same shape as the mask.
    pixel_idx = cp.zeros_like(mask, dtype=int)
    # Assign a unique integer index to each True value in the mask.
    pixel_idx[mask] = cp.arange(num_pixel)

    # Create boolean masks representing the presence of neighboring pixels in each direction.
    has_left_mask = cp.logical_and(move_right(mask), mask)
    has_right_mask = cp.logical_and(move_left(mask), mask)
    has_bottom_mask = cp.logical_and(move_top(mask), mask)
    has_top_mask = cp.logical_and(move_bottom(mask), mask)

    # Extract the horizontal and vertical components of the normal vectors for the neighboring pixels.
    nz_left = nz_horizontal[has_left_mask[mask]]
    nz_right = nz_horizontal[has_right_mask[mask]]
    nz_top = nz_vertical[has_top_mask[mask]]
    nz_bottom = nz_vertical[has_bottom_mask[mask]]

    # Create sparse matrices representing the partial derivatives for each direction.
    # top/bottom/left/right = vertical positive/vertical negative/horizontal negative/horizontal positive
    # The matrices are constructed using the extracted normal components and pixel indices.
    data = cp.stack([-nz_left/step_size, nz_left/step_size], -1).flatten()
    indices = cp.stack((pixel_idx[move_left(has_left_mask)], pixel_idx[has_left_mask]), -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_left_mask[mask].astype(int) * 2)])
    D_horizontal_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = cp.stack([-nz_right/step_size, nz_right/step_size], -1).flatten()
    indices = cp.stack((pixel_idx[has_right_mask], pixel_idx[move_right(has_right_mask)]), -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_right_mask[mask].astype(int) * 2)])
    D_horizontal_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = cp.stack([-nz_top/step_size, nz_top/step_size], -1).flatten()
    indices = cp.stack((pixel_idx[has_top_mask], pixel_idx[move_top(has_top_mask)]), -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_top_mask[mask].astype(int) * 2)])
    D_vertical_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = cp.stack([-nz_bottom/step_size, nz_bottom/step_size], -1).flatten()
    indices = cp.stack((pixel_idx[move_bottom(has_bottom_mask)], pixel_idx[has_bottom_mask]), -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_bottom_mask[mask].astype(int) * 2)])
    D_vertical_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    # Return the four sparse matrices representing the partial derivatives for each direction.
    return D_horizontal_pos, D_horizontal_neg, D_vertical_pos, D_vertical_neg


def construct_facets_from(mask):
    # Initialize an array 'idx' of the same shape as 'mask' with integers
    # representing the indices of valid pixels in the mask.
    idx = cp.zeros_like(mask, dtype=int)
    idx[mask] = cp.arange(cp.sum(mask))

    # Generate masks for neighboring pixels to define facets
    facet_move_top_mask = move_top(mask)
    facet_move_left_mask = move_left(mask)
    facet_move_top_left_mask = move_top_left(mask)

    # Identify the top-left pixel of each facet by performing a logical AND operation
    # on the masks of neighboring pixels and the input mask.
    facet_top_left_mask = facet_move_top_mask * facet_move_left_mask * facet_move_top_left_mask * mask

    # Create masks for the other three vertices of each facet by shifting the top-left mask.
    facet_top_right_mask = move_right(facet_top_left_mask)
    facet_bottom_left_mask = move_bottom(facet_top_left_mask)
    facet_bottom_right_mask = move_bottom_right(facet_top_left_mask)

    # Return a numpy array of facets by stacking the indices of the four vertices
    # of each facet along the last dimension. Each row of the resulting array represents
    # a single facet with the format [4, idx_top_left, idx_bottom_left, idx_bottom_right, idx_top_right].
    return cp.stack((4 * cp.ones(cp.sum(facet_top_left_mask).item()),
                      idx[facet_top_left_mask],
                      idx[facet_bottom_left_mask],
                      idx[facet_bottom_right_mask],
                      idx[facet_top_right_mask]), axis=-1).astype(int)


def map_depth_map_to_point_clouds(depth_map, mask, mv_mat, K=None, step_size=1, zoom=1.):
    # y
    # |  z
    # | /
    # |/
    # o ---x
    # Changed to out setting
    # y
    # |  z
    # | /
    # |/
    # o --- -x
    H, W = mask.shape
    yy, xx = cp.meshgrid(cp.arange(W), cp.arange(H))
    xx = cp.flip(xx, axis=0)

    if K is None:
        vertices = cp.ones((H, W, 4))

        x_view = (xx / (H - 1)) - 0.5  # top-to-bottom
        y_view = (yy / (W - 1)) - 0.5  # left-to-right
        
        X = y_view
        Y = x_view 
        Z = depth_map / (H - 1)

        vertices[..., 0] = X * zoom
        vertices[..., 1] = Y * zoom
        vertices[..., 2] = -Z

        vertices = vertices[mask]

        vertices = vertices @ cp.linalg.inv(mv_mat)
        vertices /= vertices[..., -1:]
        vertices = vertices[..., :-1]

    else:
        u = cp.zeros((H, W, 3))
        u[..., 0] = xx
        u[..., 1] = yy
        u[..., 2] = 1
        u = u[mask].T  # 3 x m
        vertices = (cp.linalg.inv(cp.asarray(K)) @ u).T * \
            depth_map[mask, cp.newaxis]  # m x 3

    return vertices


def sigmoid(x, k=1):
    return 1 / (1 + cp.exp(-k * x))


def bilateral_normal_integration(normal_map,
                                 normal_mask,
                                 inpaint_mask=None,
                                 k=2,
                                 lambda1=0,
                                 lambda2=0,
                                 depth_map=None,
                                 depth_mask=None,
                                 mv_mat=None,
                                 K=None,
                                 step_size=1,
                                 max_iter=150,
                                 tol=1e-4,
                                 cg_max_iter=5000,
                                 cg_tol=1e-3,
                                 save_path=None,
                                 yaw=None,
                                 pitch=None,
                                 zoom=1.,
                                 verbose=True
                                 ):
    """
    This function performs the bilateral normal integration algorithm, as described in the paper.
    It takes as input the normal map, normal mask, and several optional parameters to control the integration process.

    :param normal_map: A normal map, which is an image where each pixel's color encodes the corresponding 3D surface normal.
    :param normal_mask: A binary mask that indicates the region of interest in the normal_map to be integrated.
    :param k: A parameter that controls the stiffness of the surface.
              The smaller the k value, the smoother the surface appears (fewer discontinuities).
              If set as 0, a smooth surface is obtained (No discontinuities), and the iteration should end at step 2 since the surface will not change with iterations.

    :param depth_map: (Optional) An initial depth map to guide the integration process.
    :param depth_mask: (Optional) A binary mask that indicates the valid depths in the depth_map.

    :param lambda1 (Optional): A regularization parameter that controls the influence of the depth_map on the final result.
                               Required when depth map is input.
                               The larger the lambda1 is, the result more close to the initial depth map (fine details from the normal map are less reflected)

    :param K: (Optional) A 3x3 camera intrinsic matrix, used for perspective camera models. If not provided, the algorithm assumes an orthographic camera model.
    :param step_size: (Optional) The pixel size in the world coordinates. Default value is 1.
                                 Used only in the orthographic camera mdoel.
                                 Default value should be fine, unless you know the true value of the pixel size in the world coordinates.
                                 Do not adjust it in perspective camera model.

    :param max_iter: (Optional) The maximum number of iterations for the optimization process. Default value is 150.
                                If set as 1, a smooth surface is obtained (No discontinuities).
                                Default value should be fine.
    :param tol:  (Optional) The tolerance for the relative change in energy to determine the convergence of the optimization process. Default value is 1e-4.
                            The larger, the iteration stops faster, but the discontinuity preservation quality might be worse. (fewer discontinuities)
                            Default value should be fine.

    :param cg_max_iter: (Optional) The maximum number of iterations for the Conjugate Gradient solver. Default value is 5000.
                                   Default value should be fine.
    :param cg_tol: (Optional) The tolerance for the Conjugate Gradient solver. Default value is 1e-3.
                              Default value should be fine.

    :return: depth_map: The resulting depth map after the bilateral normal integration process.
             surface: A pyvista PolyData mesh representing the 3D surface reconstructed from the depth map.
             wu_map: A 2D image that represents the horizontal smoothness weight for each pixel. (green for smooth, blue/red for discontinuities)
             wv_map: A 2D image that represents the vertical smoothness weight for each pixel. (green for smooth, blue/red for discontinuities)
             energy_list: A list of energy values during the optimization process.
    """
    # To avoid confusion, we list the coordinate systems in this code as follows
    #
    # pixel coordinates         camera coordinates     normal coordinates (the main paper's Fig. 1 (a))
    # u                          x                              y
    # |                          |  z                           |
    # |                          | /                            o -- x
    # |                          |/                            /
    # o --- v                    o --- y                      z
    # (bottom left)
    #                       (o is the optical center;
    #                        xy-plane is parallel to the image plane;
    #                        +z is the viewing direction.)
    #
    # The input normal map should be defined in the normal coordinates.
    # The camera matrix K should be defined in the camera coordinates.
    # K = [[fx, 0,  cx],
    #      [0,  fy, cy],
    #      [0,  0,  1]]
    # I forgot why I chose the awkward coordinate system after getting used to opencv convention :(
    # but I won't touch the working code.
    
    normal_map = cp.asarray(normal_map)
    normal_mask = cp.asarray(normal_mask)
    if depth_map is not None:
        depth_map = cp.asarray(depth_map)
        depth_mask = cp.asarray(depth_mask)
    if inpaint_mask is not None:
        inpaint_mask = cp.asarray(inpaint_mask)

    num_normals = cp.sum(normal_mask).item()
    projection = "orthographic" if K is None else "perspective"
    # transfer the normal map from the normal coordinates to the camera coordinates
    nx = normal_map[normal_mask, 1]
    ny = normal_map[normal_mask, 0]
    nz = - normal_map[normal_mask, 2]
    del normal_map

    if K is not None:  # perspective
        H, W = normal_mask.shape

        yy, xx = cp.meshgrid(cp.arange(W), cp.arange(H))
        xx = cp.flip(xx, axis=0)

        cx = K[0, 2]
        cy = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        uu = xx[normal_mask] - cx
        vv = yy[normal_mask] - cy

        nz_u = uu * nx + vv * ny + fx * nz
        nz_v = uu * nx + vv * ny + fy * nz
        del xx, yy, uu, vv
    else:  # orthographic
        nz_u = nz.copy()
        nz_v = nz.copy()

    # right, left, top, bottom
    A3, A4, A1, A2 = generate_dx_dy(normal_mask, nz_horizontal=nz_v, nz_vertical=nz_u, step_size=step_size)

    pixel_idx = cp.zeros_like(normal_mask, dtype=int)
    pixel_idx[normal_mask] = cp.arange(num_normals)
    pixel_idx_flat = cp.arange(num_normals)
    pixel_idx_flat_indptr = cp.arange(num_normals + 1)

    has_left_mask = cp.logical_and(move_right(normal_mask), normal_mask)
    has_left_mask_left = move_left(has_left_mask)
    has_right_mask = cp.logical_and(move_left(normal_mask), normal_mask)
    has_right_mask_right = move_right(has_right_mask)
    has_bottom_mask = cp.logical_and(move_top(normal_mask), normal_mask)
    has_bottom_mask_bottom = move_bottom(has_bottom_mask)
    has_top_mask = cp.logical_and(move_bottom(normal_mask), normal_mask)
    has_top_mask_top = move_top(has_top_mask)

    has_left_mask_flat = has_left_mask[normal_mask]
    has_right_mask_flat = has_right_mask[normal_mask]
    has_bottom_mask_flat = has_bottom_mask[normal_mask]
    has_top_mask_flat = has_top_mask[normal_mask]

    has_left_mask_left_flat = has_left_mask_left[normal_mask]
    has_right_mask_right_flat = has_right_mask_right[normal_mask]
    has_bottom_mask_bottom_flat = has_bottom_mask_bottom[normal_mask]
    has_top_mask_top_flat = has_top_mask_top[normal_mask]

    nz_left_square = nz_v[has_left_mask_flat] ** 2
    nz_right_square = nz_v[has_right_mask_flat] ** 2
    nz_top_square = nz_u[has_top_mask_flat] ** 2
    nz_bottom_square = nz_u[has_bottom_mask_flat] ** 2

    pixel_idx_left_center = pixel_idx[has_left_mask]
    pixel_idx_right_right = pixel_idx[has_right_mask_right]
    pixel_idx_top_center = pixel_idx[has_top_mask]
    pixel_idx_bottom_bottom = pixel_idx[has_bottom_mask_bottom]

    pixel_idx_left_left_indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_left_mask_left_flat)])
    pixel_idx_right_center_indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_right_mask_flat)])
    pixel_idx_top_top_indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_top_mask_top_flat)])
    pixel_idx_bottom_center_indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_bottom_mask_flat)])

    # initialization
    wu = 0.5 * cp.ones(num_normals, float)
    wv = 0.5 * cp.ones(num_normals, float)
    # WARN: 
    z = cp.zeros(num_normals, float)

    energy = cp.sum(wu * (A1.dot(z) + nx) ** 2) + \
             cp.sum((1 - wu) * (A2.dot(z) + nx) ** 2) + \
             cp.sum(wv * (A3.dot(z) + ny) ** 2) + \
             cp.sum((1 - wv) * (A4.dot(z) + ny) ** 2)
    energy_list = []

    tic = time.time()

    energy_list = []

    if depth_map is not None:
        depth_mask_flat = depth_mask[normal_mask].astype(bool)  # shape: (num_normals,)
        z_prior = cp.log(depth_map)[normal_mask] if K is not None else depth_map[normal_mask]  # shape: (num_normals,)
        z_prior[~depth_mask_flat] = 0

    if inpaint_mask is not None:
        inv_inpaint_mask = depth_mask ^ inpaint_mask
        inv_inpaint_mask_flat = inv_inpaint_mask[normal_mask].astype(bool)  # shape: (num_normals,)

    pbar = tqdm(range(max_iter), disable=not verbose)
    depth_map_i = None
    for i in pbar:
        ################################################################################################################
        # I am manually computing A_mat = A.T @ W @ A here. It saves 2/3 time compared to the simpliest way A.T @ W @ A.
        # A.T @ W @ A can take more time than you think when the normal map become larger.
        # The diaganol matrix W=diag([wu, 1-wu, wv, 1-wv]) needs not be explicited defined in this case.
        # 
        data_term_top = wu[has_top_mask_flat] * nz_top_square
        data_term_bottom = (1 - wu[has_bottom_mask_flat]) * nz_bottom_square
        data_term_left = (1 - wv[has_left_mask_flat]) * nz_left_square
        data_term_right = wv[has_right_mask_flat] * nz_right_square

        diagonal_data_term = cp.zeros(num_normals)
        diagonal_data_term[has_left_mask_flat] += data_term_left
        diagonal_data_term[has_left_mask_left_flat] += data_term_left
        diagonal_data_term[has_right_mask_flat] += data_term_right
        diagonal_data_term[has_right_mask_right_flat] += data_term_right
        diagonal_data_term[has_top_mask_flat] += data_term_top
        diagonal_data_term[has_top_mask_top_flat] += data_term_top
        diagonal_data_term[has_bottom_mask_flat] += data_term_bottom
        diagonal_data_term[has_bottom_mask_bottom_flat] += data_term_bottom
        if depth_map is not None:
            diagonal_data_term[depth_mask_flat] += lambda1
            # Additional penalty for the inv_inpaint region
            diagonal_data_term[inv_inpaint_mask_flat] += lambda2

        A_mat_d = csr_matrix((diagonal_data_term, pixel_idx_flat, pixel_idx_flat_indptr),
                             shape=(num_normals, num_normals))

        A_mat_left_odu = csr_matrix((-data_term_left, pixel_idx_left_center, pixel_idx_left_left_indptr),
                                    shape=(num_normals, num_normals))
        A_mat_right_odu = csr_matrix((-data_term_right, pixel_idx_right_right, pixel_idx_right_center_indptr),
                                     shape=(num_normals, num_normals))
        A_mat_top_odu = csr_matrix((-data_term_top, pixel_idx_top_center, pixel_idx_top_top_indptr),
                                   shape=(num_normals, num_normals))
        A_mat_bottom_odu = csr_matrix((-data_term_bottom, pixel_idx_bottom_bottom, pixel_idx_bottom_center_indptr),
                                      shape=(num_normals, num_normals))

        A_mat_odu = A_mat_top_odu + A_mat_bottom_odu + A_mat_right_odu + A_mat_left_odu
        A_mat = A_mat_d + A_mat_odu + A_mat_odu.T  # diagnol + upper triangle + lower triangle matrix
        ################################################################################################################

        D = csr_matrix((1 / cp.clip(diagonal_data_term, 1e-5, None), pixel_idx_flat, pixel_idx_flat_indptr),
                       shape=(num_normals, num_normals))  # Jacobi preconditioner.
        b_vec = A1.T @ (wu * (-nx)) \
                + A2.T @ ((1 - wu) * (-nx)) \
                + A3.T @ (wv * (-ny)) \
                + A4.T @ ((1 - wv) * (-ny))

        if depth_map is not None:
            b_vec += lambda1 * z_prior
            b_vec[inv_inpaint_mask_flat] += lambda2 * z_prior[inv_inpaint_mask_flat]

            offset = cp.mean((z_prior - z)[depth_mask_flat])
            z = z + offset


        z, _ = cg(A_mat, b_vec, x0=z, M=D, maxiter=cg_max_iter, tol=cg_tol)

        del A_mat, b_vec, wu, wv

        # Update weights
        wu = sigmoid((A2.dot(z)) ** 2 - (A1.dot(z)) ** 2, k)  # top
        wv = sigmoid((A4.dot(z)) ** 2 - (A3.dot(z)) ** 2, k)  # right

        # Check for convergence
        energy_old = energy
        energy = cp.sum(wu * (A1.dot(z) + nx) ** 2) + \
                 cp.sum((1 - wu) * (A2.dot(z) + nx) ** 2) + \
                 cp.sum(wv * (A3.dot(z) + ny) ** 2) + \
                 cp.sum((1 - wv) * (A4.dot(z) + ny) ** 2)

        energy_list.append(energy)
        relative_energy = cp.abs(energy - energy_old) / energy_old
        pbar.set_description(
            f"step {i + 1}/{max_iter} energy: {energy:.3e}"
            f" relative energy: {relative_energy:.3e}")

        if relative_energy < tol:
    
        #if i % 100 == 0:

            depth_map_i = cp.ones_like(normal_mask, float) * cp.nan
            depth_map_i[normal_mask] = z

            # In the main paper, wu indicates the horizontal direction; wv indicates the vertical direction
            # wu_map_i = cp.ones_like(normal_mask) * cp.nan
            # wu_map_i[normal_mask] = wv

            # wv_map_i = cp.ones_like(normal_mask) * cp.nan
            # wv_map_i[normal_mask] = wu
            # 
            # combined_mask_i = ~(cp.isnan(wu_map_i) | cp.isnan(wv_map_i))

            # wu_map_i = cp.asnumpy(wu_map_i)
            # wv_map_i = cp.asnumpy(wv_map_i)

            # if inpaint_mask is not None:
            #     normal_mask_i = inpaint_mask & combined_mask_i
            # else:
            #     normal_mask_i &= combined_mask_i

            depth_map_i[~normal_mask] = cp.nan

            if K is not None:  # perspective
                depth_map = cp.exp(depth_map)
                vertices = cp.asnumpy(map_depth_map_to_point_clouds(depth_map, normal_mask, K=K))
            else:  # orthographic
                vertices = cp.asnumpy(map_depth_map_to_point_clouds(depth_map_i, normal_mask, K=None, step_size=step_size, mv_mat=mv_mat, zoom=zoom))

            facets = cp.asnumpy(construct_facets_from(normal_mask))
            if nz.mean() > 0:
                facets = facets[:, [0, 1, 4, 3, 2]]

            surface_i = pv.PolyData(vertices, facets)

            # surface_i.save(os.path.join(save_path, f"bini_{yaw:.1f}_{pitch:.1f}_i_{i:03}.ply"), binary=False)

            break

    del A1, A2, A3, A4, nx, ny
    toc = time.time()


    depth_map = cp.ones_like(normal_mask, float) * cp.nan
    depth_map[normal_mask] = z

    # In the main paper, wu indicates the horizontal direction; wv indicates the vertical direction
    wu_map = cp.ones_like(normal_mask) * cp.nan
    wu_map[normal_mask] = wv

    wv_map = cp.ones_like(normal_mask) * cp.nan
    wv_map[normal_mask] = wu

    wu_map_np = cp.asnumpy(wu_map)
    wv_map_np = cp.asnumpy(wv_map)
    
    wu_map[(wu_map < 0.4) | (wu_map > 0.6)] = cp.nan
    wv_map[(wv_map < 0.4) | (wv_map > 0.6)] = cp.nan

    combined_mask = ~(cp.isnan(wu_map) | cp.isnan(wv_map))

    combined_mask_np = cp.asnumpy(combined_mask)
    combined_mask_np = (255 * combined_mask_np).astype(np.uint8)
    morph_kernel   = np.ones((3,3), np.uint8)
    combined_mask_np = cv2.morphologyEx(combined_mask_np, cv2.MORPH_ERODE, morph_kernel, iterations=1)
    # morph_kernel   = np.ones((3,3), np.uint8)
    # normal_mask_np = cv2.morphologyEx(normal_mask_np, cv2.MORPH_OPEN,  morph_kernel, iterations=1)
    combined_mask    = cp.asarray(combined_mask_np.astype(bool))
    
    # WARN: ERODE
    # normal_mask_np = cp.asnumpy(normal_mask)
    # normal_mask_np = (255 * normal_mask_np).astype(np.uint8)
    # morph_kernel   = np.ones((3,3), np.uint8)
    # normal_mask_np = cv2.morphologyEx(normal_mask_np, cv2.MORPH_ERODE, morph_kernel, iterations=1)
    # # morph_kernel   = np.ones((3,3), np.uint8)
    # # normal_mask_np = cv2.morphologyEx(normal_mask_np, cv2.MORPH_OPEN,  morph_kernel, iterations=1)
    # normal_mask    = cp.asarray(normal_mask_np.astype(bool))

    # NOTE: Trim discontinuity?
    normal_mask &= combined_mask

    depth_map[~normal_mask] = cp.nan

    if K is not None:  # perspective
        depth_map = cp.exp(depth_map)
        vertices = cp.asnumpy(map_depth_map_to_point_clouds(depth_map, normal_mask, K=K))
    else:  # orthographic
        #vertices = cp.asnumpy(map_depth_map_to_point_clouds(depth_map, normal_mask, K=None, step_size=step_size, mv_mat=mv_mat))
        if depth_map_i is not None:
            vertices = cp.asnumpy(map_depth_map_to_point_clouds(depth_map_i, normal_mask, K=None, step_size=step_size, mv_mat=mv_mat, zoom=zoom))
        else:
            vertices = cp.asnumpy(map_depth_map_to_point_clouds(depth_map, normal_mask, K=None, step_size=step_size, mv_mat=mv_mat, zoom=zoom))

    facets = cp.asnumpy(construct_facets_from(normal_mask))
    if nz.mean() > 0:
        facets = facets[:, [0, 1, 4, 3, 2]]

    surface = pv.PolyData(vertices, facets)
    
    depth_map = cp.asnumpy(depth_map)
    normal_mask = cp.asnumpy(normal_mask)
    return depth_map, normal_mask, surface, wu_map_np, wv_map_np, energy_list

def calculate_aabb(verts):
    """
    Calculate the axis-aligned bounding box (AABB) of the vertices.
    """
    bbox_min = np.min(verts, axis=0)
    bbox_max = np.max(verts, axis=0)
    return bbox_min, bbox_max

def normalize_vertices(verts):
    """
    Normalize the vertices to fit within a unit cube centered at the origin.
    """
    bbox_min, bbox_max = calculate_aabb(verts)
    dxyz = bbox_max - bbox_min
    dist = np.sqrt(dxyz[0]**2 + dxyz[1]**2 + dxyz[2]**2)
    scale = 1.0 / dist

    # Scale the vertices
    verts = verts * scale

    # Recalculate the AABB after scaling
    bbox_min, bbox_max = calculate_aabb(verts)
    offset = -(bbox_min + bbox_max) / 2

    # Translate the vertices to center them at the origin
    verts = verts + offset
    return verts

def normalize_pyvista_mesh(mesh):
    """
    Normalize a pyvista PolyData mesh.
    """
    verts = mesh.points
    normalized_verts = normalize_vertices(verts)
    mesh.points = normalized_verts
    return mesh

def load_normal_map_with_alpha(input_data):
    if isinstance(input_data, (str, Path)):
        image = cv2.cvtColor(cv2.imread(str(input_data), cv2.IMREAD_UNCHANGED), cv2.COLOR_RGBA2BGRA)
        normal_map = image[..., :3]  # RGB channels for the normal map
        alpha_channel = image[..., 3]  # Alpha channel for the mask
    elif isinstance(input_data, Image.Image):
        arr = np.array(input_data)
        if arr.shape[-1] == 4:
            normal_map = arr[..., :3]
            alpha_channel = arr[..., 3]
        else:
            normal_map = arr
            alpha_channel = np.ones(arr.shape[:2], dtype=arr.dtype) * 255
    elif isinstance(input_data, np.ndarray):
        if input_data.shape[-1] == 4:
            normal_map = input_data[..., :3]
            alpha_channel = input_data[..., 3]
        else:
            normal_map = input_data
            alpha_channel = np.ones(input_data.shape[:2], dtype=input_data.dtype) * 255
    else:
        raise ValueError("Input must be a path, PIL Image, or numpy array.")

    if normal_map.dtype == np.uint16:
        normal_map = normal_map / 65535 * 2 - 1
    elif normal_map.dtype == np.uint8:
        normal_map = normal_map / 255 * 2 - 1

    mask = alpha_channel > 128  # Threshold alpha channel to create binary mask
    return normal_map, mask

def run_bilateral_integration(
    save_path,
    #normal_map,
    normal_input,
    # coarse_normal_path,
    normal_mask,
    depth_map,
    depth_mask,
    mv_mat,
    yaw,
    pitch,
    im_res,
    zoom=1.,
    inpaint_mask=None,
    depth_lambda=5e-3,
    depth_lambda2=5e-3,
    k=0.1,
    iters=150,
    tol=1e-4,
    cgiter=5000,
    cgtol=1e-3,
    verbose=True
):
    """Run the bilateral normal integration pipeline on given data.

    Parameters
    ----------
    path : str
        Path to the input directory containing the required data files.
    param k: A parameter that controls the stiffness of the surface.
              The smaller the k value, the smoother the surface appears (fewer discontinuities).
              If set as 0, a smooth surface is obtained (No discontinuities), and the iteration should end at step 2 since the surface will not change with iterations.
    :param iters: (Optional) The maximum number of iterations for the optimization process. Default value is 150.
                                If set as 1, a smooth surface is obtained (No discontinuities).
                                Default value should be fine.
    :param tol:  (Optional) The tolerance for the relative change in energy to determine the convergence of the optimization process. Default value is 1e-4.
                            The larger, the iteration stops faster, but the discontinuity preservation quality might be worse. (fewer discontinuities)
                            Default value should be fine.

    :param cg_max_iter: (Optional) The maximum number of iterations for the Conjugate Gradient solver. Default value is 5000.
                                   Default value should be fine.
    :param cg_tol: (Optional) The tolerance for the Conjugate Gradient solver. Default value is 1e-3.
                              Default value should be fine.
    """
    #normal_map = np.array(normal_map)
    normal_map, _ = load_normal_map_with_alpha(normal_input)
    # coarse_normal_map, _ = load_normal_map_with_alpha(coarse_normal_path)

    # normal_map = load_and_blend_maps(base_map_path=coarse_normal_path, detail_map_path=normal_path)

    # Image.fromarray((normal_mask * 255).astype(np.uint8)).save(save_path / f"normal_mask.png")
    # Image.fromarray((depth_mask * 255).astype(np.uint8)).save(save_path / f"depth_mask.png")

    # Image.fromarray((normal_mask * 255).astype(np.uint8)).save(save_path / f"normal_mask_{yaw:.1f}_{pitch:.1f}.png")

    # Update masks
    depth_mask[depth_mask < 0.5] = 0.
    normal_mask *= depth_mask.astype(bool)
    normal_map  *= depth_mask.astype(bool)[..., None]
    # Image.fromarray((normal_mask * 255).astype(np.uint8)).save(save_path / f"bini_mask_{yaw:.1f}_{pitch:.1f}.png")
    # if inpaint_mask is not None:
    #     Image.fromarray((inpaint_mask * 255).astype(np.uint8)).save(save_path / f"partial_bini_mask_{yaw:.1f}_{pitch:.1f}.png")


    # Adjust depth map
    depth_map *= -im_res
    depth_mask = normal_mask

    mv_mat = cp.asarray(mv_mat.T)

    depth_map, normal_mask, surface, wu_map, wv_map, energy_list = bilateral_normal_integration(
        normal_map=normal_map,
        normal_mask=normal_mask,
        inpaint_mask=inpaint_mask,
        depth_map=depth_map,
        depth_mask=depth_mask,
        mv_mat=mv_mat,
        lambda1=depth_lambda,
        lambda2=depth_lambda2,
        k=k,
        K=None,
        max_iter=iters,
        tol=tol,
        cg_max_iter=cgiter,
        cg_tol=cgtol,
        save_path=save_path,
        yaw=yaw,
        pitch=pitch,
        zoom=zoom,
        verbose=verbose
    )

    # Save the energy list
    # cp.save(os.path.join(save_path, f"energy_lambda1_{depth_lambda}_{yaw:.1f}_{pitch:.1f}"), cp.array(energy_list))
    # bini_surface_path = os.path.join(save_path, f"bini_mesh_k_{k}_lambda1_{depth_lambda}_{yaw:.1f}_{pitch:.1f}.ply")
    # surface.save(
    #     bini_surface_path,
    #     binary=False
    # )

    # Generate and save color maps for wu and wv
    # wu_map_img = cv2.applyColorMap((255 * wu_map).astype(np.uint8), cv2.COLORMAP_JET)
    # wv_map_img = cv2.applyColorMap((255 * wv_map).astype(np.uint8), cv2.COLORMAP_JET)
    wu_map_img = (255 * wu_map).astype(np.uint8)
    wv_map_img = (255 * wv_map).astype(np.uint8)
    wu_map_img[~normal_mask] = 255
    wv_map_img[~normal_mask] = 255

    #normal_mask = (255 * normal_mask).astype(np.uint8)
    # cv2.imwrite(os.path.join(save_path, f"wu_k_{k}_lambda1_{depth_lambda}_{yaw:.1f}_{pitch:.1f}.png"), wu_map_img)
    # cv2.imwrite(os.path.join(save_path, f"wv_k_{k}_lambda1_{depth_lambda}_{yaw:.1f}_{pitch:.1f}.png"), wv_map_img)
    # cv2.imwrite(os.path.join(save_path, f"normal_mask_k_{k}_lambda1_{depth_lambda}_{yaw:.1f}_{pitch:.1f}.png"), (normal_mask * 255).astype(np.uint8))

    # print(f"saved to {save_path}")

    return surface, normal_mask


if __name__ == '__main__':
    import cv2
    import argparse
    import os
    import warnings
    warnings.filterwarnings('ignore')
    # To ignore the possible overflow runtime warning: overflow encountered in exp return 1 / (1 + cp.exp(-k * x)).
    # This overflow issue does not affect our results as cp.exp will correctly return 0.0 when -k * x is massive.

    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise FileNotFoundError(string)

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=dir_path)
    parser.add_argument('-k', type=float, default=2)
    parser.add_argument('-i', '--iter', type=int, default=150)
    parser.add_argument('-t', '--tol', type=float, default=1e-4)
    parser.add_argument('--cgiter', type=int, default=5000)
    parser.add_argument('--cgtol', type=float, default=1e-3)
    arg = parser.parse_args()

    im_names = ['normals']

    for im_name in im_names:
        normal_map, mask = load_normal_map_with_alpha(os.path.join(os.path.abspath(arg.path), im_name+'.png'))

        if os.path.exists(os.path.join(arg.path, "render_depth.npy")):
            with open(os.path.join(arg.path, "render_depth.npy"), 'rb') as f:
                depth_map = np.load(f)
            with open(os.path.join(arg.path, "render_alpha.npy"), 'rb') as f:
                depth_mask = np.load(f)
            with open(os.path.join(arg.path, "mv_mat.npy"), 'rb') as f:
                mv_mat = np.load(f)
                mv_mat = cp.asarray(mv_mat.T)

            cos_mask = cv2.imread(os.path.join(arg.path, "cos_thresh.png"), cv2.IMREAD_GRAYSCALE) / 255.
            # depth_map = cv2.imread(os.path.join(arg.path, "init_zdepth.png"), cv2.IMREAD_GRAYSCALE) / 255.
            # depth_map = cv2.imread(os.path.join(arg.path, "init_zdepth.png"), cv2.IMREAD_GRAYSCALE)
            depth_mask[depth_mask < 0.5] = 0.
            depth_mask *= cos_mask
            mask       *= depth_mask.astype(bool)
            normal_map *= depth_mask.astype(bool)[..., None]
            depth_map  *= -1024.
            depth_mask = mask
            lambda1 = 5e-3 #1e-4


        else:
            depth_map = None
            depth_mask = None
            lambda1 = 0

        
        if os.path.exists(os.path.join(arg.path, "K.txt")):
            K = np.loadtxt(os.path.join(arg.path, "K.txt"))
            depth_map, surface, wu_map, wv_map, energy_list = bilateral_normal_integration(normal_map=normal_map,
                                                                                           normal_mask=mask,
                                                                                           k=arg.k,
                                                                                           K=K,
                                                                                           max_iter=arg.iter,
                                                                                           tol=arg.tol,
                                                                                           cg_max_iter=arg.cgiter,
                                                                                           cg_tol=arg.cgtol)
        else:
            depth_map, surface, wu_map, wv_map, energy_list = bilateral_normal_integration(normal_map=normal_map,
                                                                                           normal_mask=mask,
                                                                                           depth_map=depth_map,
                                                                                           depth_mask=depth_mask,
                                                                                           mv_mat=mv_mat,
                                                                                           lambda1=lambda1,
                                                                                           #step_size=1,
                                                                                           k=arg.k,
                                                                                           K=None,
                                                                                           max_iter=arg.iter,
                                                                                           tol=arg.tol,
                                                                                           cg_max_iter=arg.cgiter,
                                                                                           cg_tol=arg.cgtol,
                                                                                           )

        # save the resultant polygon mesh and discontinuity maps.
        cp.save(os.path.join(arg.path, "energy"), cp.array(energy_list))
        #surface = normalize_pyvista_mesh(surface)
        surface.save(os.path.join(arg.path, f"normalized_wd_fg_{im_name}_mesh_k_{arg.k}_lambda1_{lambda1}.ply"), binary=False)
        wu_map = cv2.applyColorMap(
            (255 * wu_map).astype(np.uint8), cv2.COLORMAP_JET)
        wv_map = cv2.applyColorMap(
            (255 * wv_map).astype(np.uint8), cv2.COLORMAP_JET)
        wu_map[~mask] = 255
        wv_map[~mask] = 255
        cv2.imwrite(os.path.join(arg.path, f"wu_k_{arg.k}.png"), wu_map)
        cv2.imwrite(os.path.join(arg.path, f"wv_k_{arg.k}.png"), wv_map)
