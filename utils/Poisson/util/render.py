from matplotlib import image
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

from torchvision.utils import save_image
from einops import rearrange

def _warmup(glctx):
    #windows workaround for https://github.com/NVlabs/nvdiffrast/issues/59
    def tensor(*args, **kwargs):
        return torch.tensor(*args, device='cuda', **kwargs)
    pos = tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]], dtype=torch.float32)
    tri = tensor([[0, 1, 2]], dtype=torch.int32)
    dr.rasterize(glctx, pos, tri, resolution=[256, 256])

class NormalsRenderer:
    
    _glctx: dr.RasterizeGLContext = None
    
    def __init__(
            self,
            mv: torch.Tensor,  # C,4,4
            proj: torch.Tensor,  # C,4,4
            image_size: tuple[int, int],
            ):
        self._mvp = proj @ mv  # C,4,4
        self.mv = mv  # Store the model-view matrices
        self._image_size = image_size
        self._glctx = dr.RasterizeGLContext()
        _warmup(self._glctx)

    def render(self,
               vertices: torch.Tensor,  # V,3 float
               normals: torch.Tensor,  # V,3 float
               faces: torch.Tensor,  # F,3 long
               ) -> torch.Tensor:  # C,H,W,4
        
        V = vertices.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((vertices, torch.ones(V, 1, device=vertices.device)), axis=-1)  # V,3 -> V,4
        vertices_clip = vert_hom @ self._mvp.transpose(-2, -1)  # C,V,4

        # Transform normals from world space to view space
        mv_rot = self.mv[:, :3, :3]  # C,3,3

        # Expand normals to match the number of cameras (C, V, 3)
        normals_expanded = normals.unsqueeze(0).expand(mv_rot.shape[0], -1, -1)  # Expand normals to (C, V, 3)
        
        # Transform normals to view space using mv_rot
        normals_view = torch.einsum('bij,bvj->bvi', mv_rot, normals_expanded)  # C, V, 3
        normals_view = F.normalize(normals_view, dim=-1)  # Normalize to keep them as unit vectors

        # Convert normals to color values in the range [0, 1]
        vert_col = (normals_view + 1) / 2  # Normalize to [0, 1] for color mapping
        
        rast_out, _ = dr.rasterize(self._glctx, vertices_clip.contiguous(), faces.contiguous(), resolution=self._image_size, grad_db=False)  # C,H,W,4
        
        # Ensure all inputs to interpolate are contiguous
        vert_col = vert_col.contiguous()  # C,V,3
        rast_out = rast_out.contiguous()  # C,H,W,4
        faces = faces.contiguous()  # F,3

        col, _ = dr.interpolate(vert_col, rast_out, faces)  # C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1)  # C,H,W,1
        
        col = torch.concat((col, alpha), dim=-1)  # C,H,W,4
        col = dr.antialias(col, rast_out, vertices_clip, faces)  # C,H,W,4
        
        return col  # C,H,W,4


# def render(mv,
#            proj,
#            glctx,
#            image_size,
#            vertices,
#            normals,
#            faces,
#            ):
#     
#     mvp = proj @ mv
# 
#     v = vertices.shape[0]
#     faces = faces.type(torch.int32)
#     vert_hom = torch.cat((vertices, torch.ones(v, 1, device=vertices.device)), axis=-1)  # v,3 -> v,4
#     vertices_clip = vert_hom @ mvp.transpose(-2, -1)  # c,v,4
#     vert_view     = vert_hom @ mv.transpose(-2, -1)
# 
#     # transform normals from world space to view space
#     mv_rot = mv[:, :3, :3]  # c,3,3
# 
#     # expand normals to match the number of cameras (c, v, 3)
#     normals_expanded = normals.unsqueeze(0).expand(mv_rot.shape[0], -1, -1)  # expand normals to (c, v, 3)
#     
#     # transform normals to view space using mv_rot
#     normals_view = torch.einsum('bij,bvj->bvi', mv_rot, normals_expanded)  # c, v, 3
#     normals_view = F.normalize(normals_view, dim=-1)  # normalize to keep them as unit vectors
# 
#     # convert normals to color values in the range [0, 1]
#     vert_col = (normals_view + 1) / 2  # normalize to [0, 1] for color mapping
#     
#     rast_out, _ = dr.rasterize(glctx, vertices_clip.contiguous(), faces.contiguous(), resolution=image_size, grad_db=False)  # c,h,w,4
# 
#     #depth, _ = dr.interpolate(vertices_clip[..., [2]], rast_out, faces) # [B, H, W, 1]
#     #save_image(rearrange(255. * (depth + 1.), 'b h w c -> b c h w'), './depth.png')
# 
#     #depth_view = vert_view[..., [2]]  # Extract camera-space z-values
#     depth_model = vert_hom[..., [2]]  # Extract camera-space z-values
#     depth, _ = dr.interpolate(depth_model, rast_out, faces)
# 
# 
#     
#     # ensure all inputs to interpolate are contiguous
#     vert_col = vert_col.contiguous()  # c,v,3
#     rast_out = rast_out.contiguous()  # c,h,w,4
#     faces = faces.contiguous()  # f,3
# 
#     col, _ = dr.interpolate(vert_col, rast_out, faces)  # c,h,w,3
#     alpha = torch.clamp(rast_out[..., -1:], max=1)  # c,h,w,1
#     
#     col = torch.concat((col, alpha), dim=-1)  # c,h,w,4
#     col = dr.antialias(col, rast_out, vertices_clip, faces)  # c,h,w,4
# 
#     return col, depth  # c,h,w,4

def render(mv,
           proj,
           glctx,
           image_size,
           vertices,
           normals,  # Unused in face normal mode, but kept for compatibility
           faces):
    # mv:   [c,4,4] model-view matrix per camera
    # proj: [c,4,4] projection matrix per camera
    # vertices: [v,3]
    # faces: [f,3] vertex indices
    # normals: [v,3] (Not used in face shading mode)
    # image_size: (H, W)

    # Compute MVP matrix
    mvp = proj @ mv  # [c,4,4]

    v = vertices.shape[0]
    faces = faces.type(torch.int32)
    
    # Homogeneous coordinates for vertices
    vert_hom = torch.cat((vertices, torch.ones(v, 1, device=vertices.device)), dim=-1)  # [v,4]

    # Transform vertices to clip space
    # Note: If you have multiple cameras (c > 1), you'd typically repeat vertices for each camera.
    # For simplicity, let's assume c=1 or we expand accordingly.
    # If c>1, you'd do something like:
    # vertices_clip = vert_hom.unsqueeze(0) @ mvp.transpose(-2, -1)  # [c,v,4]
    # vert_view = vert_hom.unsqueeze(0) @ mv.transpose(-2, -1)       # [c,v,4]
    # If c=1, we can expand dimensions for consistency:
    if mv.ndim == 2:
        mv = mv.unsqueeze(0)   # [1,4,4]
        mvp = mvp.unsqueeze(0) # [1,4,4]
    c = mv.shape[0]

    vert_hom_batch = vert_hom.unsqueeze(0).expand(c, -1, -1) # [c,v,4]
    vertices_clip = vert_hom_batch @ mvp.transpose(-2, -1)    # [c,v,4]
    vert_view = vert_hom_batch @ mv.transpose(-2, -1)         # [c,v,4]

    # Instead of vertex normals, compute per-face normals in view space
    # Gather per-face vertex positions
    f = faces.shape[0]

    # faces: [f,3]
    v0 = torch.gather(vert_view[..., :3], 1, faces[:, 0].view(1, -1, 1).expand(c, -1, 3).to(torch.int64))
    v1 = torch.gather(vert_view[..., :3], 1, faces[:, 1].view(1, -1, 1).expand(c, -1, 3).to(torch.int64))
    v2 = torch.gather(vert_view[..., :3], 1, faces[:, 2].view(1, -1, 1).expand(c, -1, 3).to(torch.int64))

    # Compute face normals (view space)
    face_normals_view = torch.cross((v1 - v0), (v2 - v0), dim=-1)  # [c,f,3]
    face_normals_view = F.normalize(face_normals_view, dim=-1)      # Normalize

    # Convert face normals to colors in [0,1]
    facecolor = (face_normals_view + 1.0) / 2.0  # [c,f,3]

    # Rasterize using the original faces (for geometry)
    rast_out, _ = dr.rasterize(glctx, vertices_clip.contiguous(), faces.contiguous(), resolution=image_size, grad_db=False)  # [c,h,w,4]

    t_id = rast_out[..., 3] - 1

    # Depth interpolation
    #depth_model = vert_hom_batch[..., [2]]  # [c,v,1]
    depth_view = vert_view[..., [2]]  # [c,v,1]
    depth, _ = dr.interpolate(depth_view, rast_out, faces)  # [c,h,w,1]

    # Create a triangle array that maps each face to "virtual vertices" that are all the same index
    # This causes interpolation to produce a flat result per face
    tri_facecolor = torch.arange(0, f, dtype=torch.int32, device=faces.device)[:, None].expand(-1, 3).contiguous() # [f,3]

    # Interpolate using tri_facecolor and facecolor:
    # This will produce a flat color per face, as all three "vertices" in tri_facecolor are the same index.
    col, _ = dr.interpolate(facecolor, rast_out, tri_facecolor)  # [c,h,w,3]

    # Compute alpha: pixels with a face ID >= 0 are considered covered
    alpha = torch.clamp(rast_out[..., -1:], 0, 1)  # [c,h,w,1]

    # Combine color and alpha
    col = torch.cat((col, alpha), dim=-1)  # [c,h,w,4]

    # Optional: Anti-aliasing
    col = dr.antialias(col, rast_out, vertices_clip, faces)  # [c,h,w,4]

    return col, depth #, t_id
