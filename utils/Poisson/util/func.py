import os
import torch
import io
import numpy as np
from pathlib import Path
import re
import trimesh
import imageio
import cv2
import glm

from PIL import Image
from utils.logger import get_logger

log = get_logger(__name__)

def to_numpy(*args):
    def convert(a):
        if isinstance(a,torch.Tensor):
            return a.detach().cpu().numpy()
        assert a is None or isinstance(a,np.ndarray)
        return a
    
    return convert(args[0]) if len(args)==1 else tuple(convert(a) for a in args)

def save_obj(
        vertices:torch.Tensor,
        faces:torch.Tensor,
        filename:Path
        ):
    filename = Path(filename)

    bytes_io = io.BytesIO()
    np.savetxt(bytes_io, vertices.detach().cpu().numpy(), 'v %.4f %.4f %.4f')
    np.savetxt(bytes_io, faces.cpu().numpy() + 1, 'f %d %d %d') #1-based indexing

    obj_path = filename.with_suffix('.obj')
    with open(obj_path, 'w') as file:
        file.write(bytes_io.getvalue().decode('UTF-8'))

def load_obj(
        filename:Path, 
        device='cuda'
        ) -> tuple[torch.Tensor,torch.Tensor]:
    filename = Path(filename)
    obj_path = filename.with_suffix('.obj')
    with open(obj_path) as file:
        obj_text = file.read()
    num = r"([0-9\.\-eE]+)"
    v = re.findall(f"(v {num} {num} {num})",obj_text)
    vertices = np.array(v)[:,1:].astype(np.float32)
    all_faces = []
    f = re.findall(f"(f {num} {num} {num})",obj_text)
    if f:
        all_faces.append(np.array(f)[:,1:].astype(np.long).reshape(-1,3,1)[...,:1])
    f = re.findall(f"(f {num}/{num} {num}/{num} {num}/{num})",obj_text)
    if f:
        all_faces.append(np.array(f)[:,1:].astype(np.long).reshape(-1,3,2)[...,:2])
    f = re.findall(f"(f {num}/{num}/{num} {num}/{num}/{num} {num}/{num}/{num})",obj_text)
    if f:
        all_faces.append(np.array(f)[:,1:].astype(np.long).reshape(-1,3,3)[...,:2])
    f = re.findall(f"(f {num}//{num} {num}//{num} {num}//{num})",obj_text)
    if f:
        all_faces.append(np.array(f)[:,1:].astype(np.long).reshape(-1,3,2)[...,:1])
    all_faces = np.concatenate(all_faces,axis=0)
    all_faces -= 1 #1-based indexing
    faces = all_faces[:,:,0]

    vertices = torch.tensor(vertices,dtype=torch.float32,device=device)
    faces = torch.tensor(faces,dtype=torch.long,device=device)

    return vertices,faces

def save_ply(
        filename:Path,
        vertices:torch.Tensor, #V,3
        faces:torch.Tensor, #F,3
        vertex_colors:torch.Tensor=None, #V,3
        vertex_normals:torch.Tensor=None, #V,3
        ):
        
    filename = Path(filename).with_suffix('.ply')
    vertices,faces,vertex_colors = to_numpy(vertices,faces,vertex_colors)
    assert np.all(np.isfinite(vertices)) and faces.min()==0 and faces.max()==vertices.shape[0]-1

    header = 'ply\nformat ascii 1.0\n'

    header += 'element vertex ' + str(vertices.shape[0]) + '\n'
    header += 'property double x\n'
    header += 'property double y\n'
    header += 'property double z\n'

    if vertex_normals is not None:
        header += 'property double nx\n'
        header += 'property double ny\n'
        header += 'property double nz\n'

    if vertex_colors is not None:
        assert vertex_colors.shape[0] == vertices.shape[0]
        color = (vertex_colors*255).astype(np.uint8)
        header += 'property uchar red\n'
        header += 'property uchar green\n'
        header += 'property uchar blue\n'

    header += 'element face ' + str(faces.shape[0]) + '\n'
    header += 'property list int int vertex_indices\n'
    header += 'end_header\n'

    with open(filename, 'w') as file:
        file.write(header)

        for i in range(vertices.shape[0]):
            s = f"{vertices[i,0]} {vertices[i,1]} {vertices[i,2]}"    
            if vertex_normals is not None:
                s += f" {vertex_normals[i,0]} {vertex_normals[i,1]} {vertex_normals[i,2]}"
            if vertex_colors is not None:
                s += f" {color[i,0]:03d} {color[i,1]:03d} {color[i,2]:03d}"
            file.write(s+'\n')
        
        for i in range(faces.shape[0]):
            file.write(f"3 {faces[i,0]} {faces[i,1]} {faces[i,2]}\n")
    full_verts = vertices[faces] #F,3,3

def save_single_image(
        images:torch.Tensor, #B,H,W,CH
        dir:Path,
        img_name
        ):
    dir = Path(dir)
    dir.mkdir(parents=True,exist_ok=True)
    Image.fromarray((images.detach()[0,:,:,:3]*255).squeeze().clamp(max=255).type(torch.uint8).cpu().numpy()).save(os.path.join(dir, img_name))
    
def save_images(
        images:torch.Tensor, #B,H,W,CH
        dir:Path,
        ):
    dir = Path(dir)
    dir.mkdir(parents=True,exist_ok=True)
    for i in range(images.shape[0]):
        log.info((images.detach()[i,:,:,:3]*255).clamp(max=255).type(torch.uint8).cpu().numpy().shape)
        #cv2.imwrite(os.path.join(dir, f'{i:02d}.png'), (images.detach()[i,:,:,:3]*255).clamp(max=255).type(torch.uint8).cpu().numpy())
        Image.fromarray((images.detach()[i,:,:,:3]*255).squeeze().clamp(max=255).type(torch.uint8).cpu().numpy()).save(os.path.join(dir, f'{i:02d}.png'))


def normalize_vertices(
        vertices:torch.Tensor, #V,3
    ):
    """shift and resize mesh to fit into a unit sphere"""
    vertices -= (vertices.min(dim=0)[0] + vertices.max(dim=0)[0]) / 2
    vertices /= torch.norm(vertices, dim=-1).max()
    return vertices

def laplacian(
        num_verts:int,
        edges: torch.Tensor #E,2
        ) -> torch.Tensor: #sparse V,V
    """create sparse Laplacian matrix"""
    V = num_verts
    E = edges.shape[0]

    #adjacency matrix,
    idx = torch.cat([edges, edges.fliplr()], dim=0).type(torch.long).T  # (2, 2*E)
    ones = torch.ones(2*E, dtype=torch.float32, device=edges.device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    #degree matrix
    deg = torch.sparse.sum(A, dim=1).to_dense()
    idx = torch.arange(V, device=edges.device)
    idx = torch.stack([idx, idx], dim=0)
    D = torch.sparse.FloatTensor(idx, deg, (V, V))

    return D - A

def _translation(x, y, z, device):
    return torch.tensor([[1., 0, 0, x],
                    [0, 1, 0, y],
                    [0, 0, 1, z],
                    [0, 0, 0, 1]],device=device) #4,4

def _projection(r, device, l=None, t=None, b=None, n=1.0, f=50.0, flip_y=True):
    if l is None:
        l = -r
    if t is None:
        t = r
    if b is None:
        b = -t
    p = torch.zeros([4,4],device=device)
    p[0,0] = 2*n/(r-l)
    p[0,2] = (r+l)/(r-l)
    p[1,1] = 2*n/(t-b) * (-1 if flip_y else 1)
    p[1,2] = (t+b)/(t-b)
    p[2,2] = -(f+n)/(f-n)
    p[2,3] = -(2*f*n)/(f-n)
    p[3,2] = -1
    return p #4,4

def ortho_proj(device, near=0.0001, far=1000., max_x=0.5, min_x=-0.5, max_y=0.5, min_y=-0.5):
    ortho_matrix = np.array([
        [2 / (max_x - min_x), 0, 0, -(max_x + min_x) / (max_x - min_x)],
        [0, -2 / (max_y - min_y), 0, -(max_y + min_y) / (max_y - min_y)],  # Note the negative sign
        [0, 0, -2 / (far - near), -(far + near) / (far - near)],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    ortho_matrix = torch.from_numpy(ortho_matrix).to(device=device)
    return ortho_matrix

def make_star_cameras(az_count,pol_count,distance:float=10.,r=None,image_size=[512,512],device='cuda'):
    if r is None:
        r = 1/distance
    A = az_count
    P = pol_count
    C = A * P

    phi = torch.arange(0,A) * (2*torch.pi/A)
    phi_rot = torch.eye(3,device=device)[None,None].expand(A,1,3,3).clone()
    phi_rot[:,0,2,2] = phi.cos()
    phi_rot[:,0,2,0] = -phi.sin()
    phi_rot[:,0,0,2] = phi.sin()
    phi_rot[:,0,0,0] = phi.cos()
    
    theta = torch.arange(1,P+1) * (torch.pi/(P+1)) - torch.pi/2
    theta_rot = torch.eye(3,device=device)[None,None].expand(1,P,3,3).clone()
    theta_rot[0,:,1,1] = theta.cos()
    theta_rot[0,:,1,2] = -theta.sin()
    theta_rot[0,:,2,1] = theta.sin()
    theta_rot[0,:,2,2] = theta.cos()

    mv = torch.empty((C,4,4), device=device)
    mv[:] = torch.eye(4, device=device)
    mv[:,:3,:3] = (theta_rot @ phi_rot).reshape(C,3,3)
    mv = _translation(0, 0, -distance, device) @ mv

    return mv, _projection(r,device)

def make_star_ortho_cameras(az_count,pol_count,distance:float=10.,r=None, image_size=[512,512], device='cuda'):
    if r is None:
        r = 1/distance
    A = az_count
    P = pol_count
    C = A * P

    phi = torch.arange(0,A) * (2*torch.pi/A)
    phi_rot = torch.eye(3,device=device)[None,None].expand(A,1,3,3).clone()
    phi_rot[:,0,2,2] = phi.cos()
    phi_rot[:,0,2,0] = -phi.sin()
    phi_rot[:,0,0,2] = phi.sin()
    phi_rot[:,0,0,0] = phi.cos()
    
    #theta = torch.arange(1,P+1) * (torch.pi/(P+1)) - torch.pi/2
    theta = torch.tensor([torch.pi * 2.], device=device)
    theta_rot = torch.eye(3,device=device)[None,None].expand(1,P,3,3).clone()
    theta_rot[0,:,1,1] = theta.cos()
    theta_rot[0,:,1,2] = -theta.sin()
    theta_rot[0,:,2,1] = theta.sin()
    theta_rot[0,:,2,2] = theta.cos()

    mv = torch.empty((C,4,4), device=device)
    mv[:] = torch.eye(4, device=device)
    mv[:,:3,:3] = (theta_rot @ phi_rot).reshape(C,3,3)
    mv = _translation(0, 0, -distance, device) @ mv

    return mv, ortho_proj(device, max_x=1, min_x=-1, max_y=1, min_y=-1)

def make_era_ortho_cameras(r=1., device='cuda'):

    #phi = torch.tensor([0, -torch.pi/2, torch.pi/2, torch.pi], dtype=torch.float32) 

    phi = torch.tensor([0+ torch.pi, -torch.pi/2+ torch.pi, torch.pi/2+ torch.pi, torch.pi+ torch.pi], dtype=torch.float32) 
 

    #phi = torch.tensor([0, -torch.pi/4, torch.pi/4, -torch.pi/2, torch.pi/2, torch.pi], dtype=torch.float32) 
    #phi = torch.tensor([0.0000, 1.5708, 3.1416, 4.7124], dtype=torch.float32)
    A = len(phi)
    C = A  # Since we only have one polar angle

    phi_rot = torch.eye(3,device=device)[None,None].expand(A,1,3,3).clone()
    phi_rot[:,0,2,2] = phi.cos()
    phi_rot[:,0,2,0] = -phi.sin()
    phi_rot[:,0,0,2] = phi.sin()
    phi_rot[:,0,0,0] = phi.cos()
    
    theta = torch.tensor([torch.pi * 2.], device=device)
    theta_rot = torch.eye(3,device=device)[None,None].expand(1,1,3,3).clone()
    theta_rot[0,:,1,1] = theta.cos()
    theta_rot[0,:,1,2] = -theta.sin()
    theta_rot[0,:,2,1] = theta.sin()
    theta_rot[0,:,2,2] = theta.cos()

    mv = torch.empty((C,4,4), device=device)
    mv[:] = torch.eye(4, device=device)
    mv[:,:3,:3] = (theta_rot @ phi_rot).reshape(C,3,3)
    mv = _translation(0, 0, -r, device) @ mv

    return mv, ortho_proj(device, max_x=1, min_x=-1, max_y=1, min_y=-1)

def make_sphere(level:int=2,radius=1.,device='cuda') -> tuple[torch.Tensor,torch.Tensor]:
    sphere = trimesh.creation.icosphere(subdivisions=level, radius=1.0, color=None)
    vertices = torch.tensor(sphere.vertices, device=device, dtype=torch.float32) * radius
    faces = torch.tensor(sphere.faces, device=device, dtype=torch.long)
    return vertices,faces

# def make_yaw_pitch_ortho_cameras(pitchs, yaws, r=1., device='cuda'):
# 
#     #phi = torch.tensor([0+ torch.pi, -torch.pi/2+ torch.pi, torch.pi/2+ torch.pi, torch.pi+ torch.pi], dtype=torch.float32) 
#     A = len(yaws)
#     C = A  # Since we only have one polar angle
# 
#     phi_rot = torch.eye(3,device=device)[None,None].expand(A,1,3,3).clone()
#     phi_rot[:,0,2,2] =  yaws.cos()
#     phi_rot[:,0,2,0] =  yaws.sin()
#     phi_rot[:,0,0,2] = -yaws.sin()
#     phi_rot[:,0,0,0] =  yaws.cos()
#     
#     #theta = torch.tensor([torch.pi * 2.], device=device)
#     theta_rot = torch.eye(3,device=device)[None,None].expand(1,1,3,3).clone()
#     theta_rot[0,:,1,1] =  pitchs.cos()
#     theta_rot[0,:,1,2] = -pitchs.sin()
#     theta_rot[0,:,2,1] =  pitchs.sin()
#     theta_rot[0,:,2,2] =  pitchs.cos()
# 
#     mv = torch.empty((C,4,4), device=device)
#     mv[:] = torch.eye(4, device=device)
#     mv[:,:3,:3] = (theta_rot @ phi_rot).reshape(C,3,3)
#     mv = _translation(0, 0, -r, device) @ mv
# 
#     return mv #, _ortho_proj(device, max_x=1, min_x=-1, max_y=1, min_y=-1)

def make_yaw_pitch_ortho_cameras(pitchs, yaws, r=1., device='cuda'):
    A = len(yaws)
    C = A  # Since we want to handle one camera per yaw-pitch pair

    # Create yaw rotation matrices (phi_rot)
    phi_rot = torch.eye(3, device=device)[None, :].expand(A, 3, 3).clone()
    phi_rot[:, 2, 2] = yaws.cos()
    phi_rot[:, 2, 0] = yaws.sin()
    phi_rot[:, 0, 2] = -yaws.sin()
    phi_rot[:, 0, 0] = yaws.cos()
    
    # Create pitch rotation matrices (theta_rot)
    theta_rot = torch.eye(3, device=device)[None, :].expand(A, 3, 3).clone()
    theta_rot[:, 1, 1] = pitchs.cos()
    theta_rot[:, 1, 2] = -pitchs.sin()
    theta_rot[:, 2, 1] = pitchs.sin()
    theta_rot[:, 2, 2] = pitchs.cos()

    # Combine rotations for each yaw-pitch pair
    rot_matrix = (theta_rot @ phi_rot)

    # Initialize the model-view matrix (mv)
    mv = torch.eye(4, device=device).expand(C, 4, 4).clone()
    mv[:, :3, :3] = rot_matrix

    # Apply translation
    mv = _translation(0, 0, -r, device) @ mv

    return mv


# def make_pitch_yaw_ortho_cameras(pitches, yaws, r=1.0, device='cuda'):
#     A = len(pitches)  # Number of cameras
#     
#     # Create empty tensors for rotation matrices
#     mv = torch.empty((A, 4, 4), device=device)
#     mv[:] = torch.eye(4, device=device)
#     
#     # Compute rotation matrices based on pitch and yaw
#     for i, (pitch, yaw) in enumerate(zip(pitches, yaws)):
#         # Yaw rotation around Y-axis
#         yaw_rot = torch.eye(3, device=device)
#         yaw_rot[0, 0] = yaw.cos()
#         yaw_rot[0, 2] = yaw.sin()
#         yaw_rot[2, 0] = -yaw.sin()
#         yaw_rot[2, 2] = yaw.cos()
#         
#         # Pitch rotation around X-axis
#         pitch_rot = torch.eye(3, device=device)
#         pitch_rot[1, 1] = pitch.cos()
#         pitch_rot[1, 2] = pitch.sin()
#         pitch_rot[2, 1] = -pitch.sin()
#         pitch_rot[2, 2] = pitch.cos()
#         
#         # Combined rotation matrix
#         rotation_matrix = yaw_rot @ pitch_rot
#         
#         # Update the rotation part of the model-view matrix
#         mv[i, :3, :3] = rotation_matrix
#         # Apply translation to the camera (moving back by radius r)
#         mv[i] = _translation(0, 0, -r, device) @ mv[i]
#     
#     return mv, _ortho_proj(device, max_x=1, min_x=-1, max_y=1, min_y=-1)

def safe_normalize(v, eps=1e-9):
    norm = torch.norm(v)
    if norm < eps:
        return v
    return v / norm

def make_pitch_yaw_ortho_cameras(pitches, yaws, r=1., device='cuda'):
    A = len(pitches)  # Number of cameras
    
    # Create empty tensors for view matrices
    mv = torch.empty((A, 4, 4), device=device)
    
    # Compute view matrices based on pitch and yaw
    for i, (pitch, yaw) in enumerate(zip(pitches, yaws)):
        # Calculate the camera position
        camera_x = r * torch.cos(pitch) * torch.sin(yaw)
        camera_y = -r * torch.sin(pitch)
        camera_z = r * torch.cos(pitch) * torch.cos(yaw)

        camera_position = torch.tensor([camera_x, camera_y, camera_z], device=device, dtype=torch.float32)
        target = torch.tensor([0, 0, 0], device=device, dtype=torch.float32)
        up = torch.tensor([0, 1, 0], device=device, dtype=torch.float32)
        
        # Compute the forward, right, and up vectors
        forward = safe_normalize(target - camera_position)
        right = safe_normalize(torch.cross(up, forward))
        up = torch.cross(forward, right)  # Recompute up to ensure orthogonality

        # Construct the view matrix
        view_matrix = torch.eye(4, device=device, dtype=torch.float32)
        view_matrix[0, :3] = right
        view_matrix[1, :3] = up
        view_matrix[2, :3] = -forward
        view_matrix[:3, 3] = -torch.matmul(view_matrix[:3, :3], camera_position)

        mv[i] = view_matrix
    
    # Orthographic projection matrix
    near = 0.0001
    far = 1000.0
    left = -1.0
    right = 1.0
    bottom = -1.0
    top = 1.0

    ortho_proj = torch.tensor([
        [2 / (right - left), 0, 0, -(right + left) / (right - left)],
        [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
        [0, 0, -2 / (far - near), -(far + near) / (far - near)],
        [0, 0, 0, 1]
    ], device=device, dtype=torch.float32)
    
    return mv, ortho_proj
