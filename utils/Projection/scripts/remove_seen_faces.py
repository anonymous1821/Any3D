import os
import math
import time

from pathlib import Path
from PIL import Image

import moderngl
import numpy as np
import cupy as cp
import glm

from utils.Poisson.mesh import Mesh
from utils.Projection.utils.cam_utils import get_view_projection_matrices
from utils.logger import get_logger

log = get_logger(__name__)

def calculate_camera_position(pitch, yaw):
    pitch += 180
    pitch_rad = glm.radians(pitch)
    yaw_rad   = glm.radians(yaw)
    camera_x  = glm.cos(pitch_rad) * glm.sin(yaw_rad)
    camera_y  = -glm.sin(pitch_rad)
    camera_z  = glm.cos(pitch_rad) * glm.cos(yaw_rad)
    return glm.vec3(camera_x, camera_y, camera_z)

def remove_shared_verts(mesh):
    vertices  = mesh.v.detach().cpu().numpy()
    triangles = mesh.f.detach().cpu().numpy()

    # Flatten the vertices so that each face has unique vertices
    new_vertices = vertices[triangles].reshape(-1, 3)
    new_triangles = np.arange(len(triangles) * 3).reshape(-1, 3)

    mesh.v = new_vertices
    mesh.f = new_triangles
    mesh.vc = np.zeros_like(mesh.v)

    return mesh

def remove_outside_silhouette_faces(obj_path, partial_meshes_path, camera_plan, textures, target_angle):
    # Load the mesh
    mesh = Mesh().load(obj_path, resize=False)
    no_dup_mesh = remove_shared_verts(mesh)

    vertices = no_dup_mesh.v.astype('f4')
    faces    = no_dup_mesh.f.astype('i4')

    # NOTE: std430 layout, need to pad to have 16 bytes
    vertices_padded = np.zeros((len(vertices), 4), dtype='f4')
    vertices_padded[:, :3] = vertices

    faces_padded = np.zeros((len(faces), 4), dtype='i4')
    faces_padded[:, :3] = faces 

    # Set up OpenGL context
    ctx = moderngl.create_standalone_context(require=430, backend='egl')

    # Create buffer objects
    vertex_buffer = ctx.buffer(vertices_padded.tobytes())
    face_buffer   = ctx.buffer(faces_padded.tobytes())

    # Prepare face flags buffer (0: remove, 1: keep)
    face_flags = np.zeros(len(faces), dtype='i4')
    face_flags_buffer = ctx.buffer(face_flags.tobytes())

    # Read and compile the compute shader
    shader_path = 'Projection/shaders/silhouette_shader.glsl'
    with open(shader_path, 'r') as f:
        compute_shader_source = f.read()
    try:
        compute_shader = ctx.compute_shader(source=compute_shader_source)
    except moderngl.Error as e:
        log.error("Shader compilation failed:")
        log.error(e)
        return
    
    width, height = textures[0].size  # Assuming all masks are the same size

    # Convert the silhouette mask to a binary image and upload it as a texture
    silhouette_textures = []
    for silhouette_mask in textures:
        silhouette_mask = silhouette_mask.split()[-1] 
        silhouette_data = np.array(silhouette_mask.convert('L')).astype('u1')
        silhouette_texture = ctx.texture((width, height), 1, silhouette_data.tobytes(), dtype='u1')
        silhouette_textures.append(silhouette_texture)

    # Create the projection output texture
    # We'll use an RGBA8 texture, but we'll write only to the red channel
    projection_output = ctx.texture(
        (width, height),
        4,  # RGBA channels
        data=np.zeros((height, width, 4), dtype='u1').tobytes(),
        dtype='u1'
    )
    # projection_output.clear(0.0, 0.0, 0.0, 0.0)  # Initialize to black
    projection_output.bind_to_image(4, read=False, write=True)

    all_remove_faces = set()

    for idx, (target_view, texture) in enumerate(zip(camera_plan, textures)):
        t_yaw = target_view.yaw
        t_pitch = target_view.pitch

        camera_distance = 1.5
        view_matrix, projection_matrix = get_view_projection_matrices(camera_distance, t_pitch+180, t_yaw)

        mvp_matrix = projection_matrix * view_matrix * glm.mat4(1.0)  # Assuming model matrix is identity

        # Write uniforms to shader
        compute_shader['mvp_matrix'].write(np.array(mvp_matrix).astype('f4').tobytes())
        compute_shader['image_size'].value = (width, height)

        # Bind buffers
        vertex_buffer.bind_to_storage_buffer(binding=0)
        face_buffer.bind_to_storage_buffer(binding=1)
        face_flags_buffer.bind_to_storage_buffer(binding=2)

        # Bind the silhouette texture
        silhouette_textures[idx].use(location=3)

        # Dispatch compute shader
        num_faces = len(faces)
        workgroup_size = 256  # Adjust based on your GPU's capabilities
        num_workgroups = math.ceil(num_faces / workgroup_size)
        compute_shader.run(num_workgroups, 1, 1)

        # Read back the face flags
        face_flags_buffer.read_into(face_flags)
        face_flags_np = np.frombuffer(face_flags, dtype=np.int32)

        log.info(f"Removed faces from view {idx}")

        # Update faces to remove
        remove_face_indices = np.where(face_flags_np == 0)[0]
        all_remove_faces.update(remove_face_indices.tolist())

    # Remove faces outside the silhouette
    all_faces = set(range(len(faces)))
    faces_to_keep = np.array(list(all_faces - all_remove_faces))

    #faces_to_keep = np.array(list(all_remove_faces))
    new_faces = faces[faces_to_keep]

    new_vertices = vertices[new_faces].reshape(-1, 3)
    new_faces = np.arange(len(new_vertices)).reshape(-1, 3)

    # Save the new mesh
    output_obj_fp = os.path.join(partial_meshes_path, f'silhouette_based_removal_{target_angle.yaw:.1f}_{target_angle.pitch:.1f}.obj')
    Mesh.write_obj_with_only_verts_and_faces(output_obj_fp, new_vertices, new_faces)

    # Clean up
    vertex_buffer.release()
    face_buffer.release()
    face_flags_buffer.release()
    for texture in silhouette_textures:
        texture.release()
    compute_shader.release()
    ctx.release()

def remove_visible_faces(obj_path, partial_meshes_path, camera_plan, textures, target_angle, dot_threshold=0.3):
    # Load the mesh
    mesh = Mesh().load(obj_path, resize=False)
    no_dup_mesh = remove_shared_verts(mesh)

    vertices = no_dup_mesh.v.astype('f4')
    faces    = no_dup_mesh.f.astype('i4').flatten()

    # Set up OpenGL context
    ctx = moderngl.create_standalone_context(require=430, backend='egl')

    # Create buffer objects
    vbo = ctx.buffer(vertices.tobytes())
    ibo = ctx.buffer(faces.tobytes())

    # Prepare face flags buffer (0: remove, 1: keep)
    face_flags = np.zeros(len(faces), dtype='i4')
    face_flags_buffer = ctx.buffer(face_flags.tobytes())

    # Load shaders
    prog = ctx.program(
        vertex_shader='''
                #version 330 core
                layout(location = 0) in vec3 in_vert;
                flat out int face_id;
                
                uniform mat4 model;
                uniform mat4 view;
                uniform mat4 projection;
                
                void main() {
                    gl_Position = projection * view * model * vec4(in_vert, 1.0);
                    face_id = gl_VertexID / 3;
                }
            ''',
        fragment_shader='''
                    #version 330 core
                    flat in int face_id;
                    out int out_face_id;
                    
                    void main() {
                        out_face_id = face_id;
                    }
                '''
    )

    vao = ctx.vertex_array(prog, [(vbo, '3f', 'in_vert')], index_buffer=ibo)

    # Create framebuffer with an integer texture
    width, height = textures[0].size
    face_id_texture = ctx.texture((width, height), 1, dtype='i4')
    face_id_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
    depth_buffer = ctx.depth_renderbuffer((width, height))
    framebuffer = ctx.framebuffer([face_id_texture], depth_buffer)

    all_remove_faces = set()

    for target_view in camera_plan:
        t_yaw   = target_view.yaw
        t_pitch = target_view.pitch
        t_zoom  = target_view.zoom

        camera_distance = 1.5
        view_matrix, projection_matrix = get_view_projection_matrices(camera_distance, t_pitch+180, t_yaw, t_zoom)

        # Write matrices to shader
        prog['projection'].write(projection_matrix)
        prog['view'].write(view_matrix)
        prog['model'].write(glm.mat4(1.0))  # No rotation of the object itself

        # Render to framebuffer
        #NOTE: Modified
        framebuffer.use()
        ctx.clear(-1, -1, -1, -1)
        ctx.enable(moderngl.DEPTH_TEST)
        vao.render()

        # Read the face ID texture
        face_ids = np.frombuffer(face_id_texture.read(), dtype=np.int32)
        face_ids = face_ids.reshape((height, width))

        visible_faces = np.unique(face_ids)
        # Exclude background (-1) and ensure indices are within bounds
        # Note: We found cases where face_id could be out of bounds (e.g. 32512 vs 10804 faces)
        visible_faces = visible_faces[(visible_faces >= 0) & (visible_faces < len(no_dup_mesh.f))]

        cam_dir = calculate_camera_position(t_pitch, t_yaw)

        face_normals_cp      = cp.asarray(mesh.face_normals)
        camera_dir_cp        = cp.asarray(cam_dir)
        visible_faces_cp     = cp.asarray(visible_faces, dtype=cp.int32)
        normals_subset_cp    = face_normals_cp[visible_faces_cp]
        dot_vals_cp          = cp.sum(normals_subset_cp * camera_dir_cp, axis=1)
        keep_mask_cp         = cp.abs(dot_vals_cp) >= dot_threshold
        remove_candidates_cp = visible_faces_cp[keep_mask_cp]
        remove_candidates    = remove_candidates_cp.get().tolist()

        all_remove_faces.update(remove_candidates)

    # Process invisible vertices
    visible_faces_list = list(all_remove_faces)
    visible_verts_idx = no_dup_mesh.f[visible_faces_list].flatten()
    vert_mask = np.ones(len(vertices), dtype=bool)
    vert_mask[visible_verts_idx] = False

    invisible_verts = vertices[vert_mask]
    invisible_faces = np.arange(len(invisible_verts)).reshape(-1, 3)

    # Save invisible mesh
    partial_obj_fp = os.path.join(partial_meshes_path, f'visibility_based_removal_{target_angle.yaw:.1f}_{target_angle.pitch:.1f}.obj')
    Mesh.write_obj_with_only_verts_and_faces(partial_obj_fp, invisible_verts, invisible_faces)

    # Clean up
    vbo.release()
    ibo.release()
    vao.release()
    prog.release()
    face_id_texture.release()
    depth_buffer.release()
    framebuffer.release()
    ctx.release()

    return partial_obj_fp

def remove_seen_faces(
        obj_path, 
        partial_meshes_dir, 
        texture_dir,
        seen_yaw_pitch_list,
        target_angle,
        dot_threshold,
    ):
    width  = 1024 * 4
    height = 1024 * 4

    # Create the output directory if it doesn't exist.
    Path(partial_meshes_dir).mkdir(parents=True, exist_ok=True)

    textures = []
    for angle in seen_yaw_pitch_list:
        # Build a pattern that matches the unique texture file including the zoom parameter.
        pattern = f"refined_{angle.yaw:.1f}_{angle.pitch:.1f}_*.png"
        candidate_files = list(Path(texture_dir).glob(pattern))
        
        if not candidate_files:
            # If no texture exists for the given yaw and pitch, skip it.
            breakpoint()
            continue
        
        # Since it's guaranteed to be unique, take the first (and only) file.
        file_path = candidate_files[0]
        
        # Extract the zoom value from the filename.
        # Assuming the filename is "refined_{yaw}_{pitch}_{zoom}.png", split by '_' and take the last element of the stem.
        # e.g., for "refined_45.0_30.0_1.234567890123456.png"
        zoom_str = file_path.stem.split('_')[-1]
        try:
            zoom = float(zoom_str)
        except ValueError:
            raise ValueError(f"Unable to convert zoom value {zoom_str} to float for file {file_path}")
        
        # Append the zoom value to the angle.
        angle.zoom = zoom
        
        # Open the image file, convert to RGBA, and resize.
        im = (
            Image.open(file_path)
            .convert('RGBA')
            .resize(size=(width, height), resample=Image.Resampling.NEAREST)
        )
        textures.append(im)

    # Here camera_plan remains the same, but now every angle is expected to have an attribute 'zoom'
    camera_plan = seen_yaw_pitch_list

    # Call the function that removes visible faces.
    vis_mesh_path = remove_visible_faces(
        obj_path, 
        partial_meshes_dir, 
        camera_plan, 
        textures, 
        target_angle, 
        dot_threshold
    )

    return vis_mesh_path
