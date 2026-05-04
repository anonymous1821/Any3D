import os
from pathlib import Path

import moderngl
import numpy as np
import glm

import cupy as cp
from cupyx.scipy.ndimage import convolve

from PIL import Image
from objloader import Obj

from .shaders.generate_shader import generate_fragment_shader_baking_code
from utils.logger import get_logger
log = get_logger(__name__)
def recompute_normals(vertices, faces):
    normals = np.zeros(vertices.shape, dtype=vertices.dtype)
    tris = vertices[faces]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n = n / np.linalg.norm(n, axis=1)[:, np.newaxis]
    for i in range(3):
        normals[faces[:, i]] += n
    normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
    return normals

class HeadlessBaker:
    def __init__(self, bake_vertex_shader_path, obj_mesh, texture_dir, device_idx):
        # Context creation
        self.ctx = moderngl.create_context(standalone=True, backend='egl', device_index=device_idx)
        self.ctx.gc_mode = 'auto'


        obj = obj_mesh #Obj.open(obj_path)

        # Load multiple textures and create a texture array
        texture_files = list(Path(texture_dir).glob('*.png'))
        textures = [Image.open(file).convert('RGBA') for file in texture_files]
        texture_count = len(textures)

        self.refined_textures = np.zeros(texture_count, dtype='int32')
        for idx, file in enumerate(texture_files):
            if file.parts[-1].split(sep='_')[0] == 'refined':
                self.refined_textures[idx] = 1
            else:
                self.refined_textures[idx] = 0

        # Generate the fragment shader code dynamically
        fragment_bake_shader_code   = generate_fragment_shader_baking_code(texture_count)

        self.depth_prog = self.ctx.program(
            vertex_shader='''
                #version 330 core
                layout(location = 0) in vec3 in_vert;
                
                uniform mat4 model;
                uniform mat4 view;
                uniform mat4 projection;
                
                out vec3 frag_position;
                
                void main() {
                    vec4 pos_world = model * vec4(in_vert, 1.0);
                    frag_position = pos_world.xyz; // Pass world position to fragment shader
                    gl_Position = projection * view * pos_world;
                }
            ''',
            fragment_shader='''
                #version 330 core
                in vec3 frag_position;
                
                uniform mat4 view;
                uniform mat4 projection;
                
                out float frag_depth;
                
                void main() {
                    // Transform frag_position using view and projection matrices
                    vec4 pos_view = view * vec4(frag_position, 1.0);
                    vec4 pos_clip = projection * pos_view;
                    vec3 pos_ndc = pos_clip.xyz / pos_clip.w;
                    frag_depth = pos_ndc.z; // Compute depth in NDC space
                }
            '''
        )
        self.bake_prog = self.ctx.program(
            vertex_shader=open(bake_vertex_shader_path).read(),
            fragment_shader=fragment_bake_shader_code
        )

        self.camera_distance = 1.5 # Initial distance of the camera from the object
        self.epsilon = 0.001

        # Toggles for alpha blending
        self.enable_alpha_blending = False
        # Track which textures are active (default to all active)
        self.active_textures = np.ones(texture_count, dtype='int32')
        # Properly set the active_textures uniform array in the shader
        # self.bake_prog['active_textures'].write(self.active_textures.tobytes())
        # self.bake_prog['refined_textures'].write(self.refined_textures.tobytes())
        self.bake_prog['epsilon'].value = self.epsilon

        width, height = textures[0].size
        texture_array_data = np.array([np.array(tex) for tex in textures], dtype='uint8')
        self.texture_array = self.ctx.texture_array((width, height, texture_count), 4, texture_array_data)

        # Bind textures and set uniforms
        self.bake_prog['textures'] = 0
        # self.bake_prog['texture_count'].value = texture_count
        self.texture_array.use(location=0)

        # Create buffers and VAO for the mesh
        self.vbo = self.ctx.buffer(obj.pack('vx vy vz nx ny nz tx ty'))
        self.vao = self.ctx.vertex_array(self.bake_prog, [
            (self.vbo, '3f 3f 2f', 'in_vert', 'in_norm', 'in_uv')
        ])

        self.vbo_depth = self.ctx.buffer(obj.pack('vx vy vz'))
        self.depth_vao = self.ctx.vertex_array(self.depth_prog, [
            (self.vbo_depth, '3f', 'in_vert')
        ])

        # Define projector directions dynamically based on texture filenames
        self.projector_directions = self.calculate_projector_directions(texture_files)

        pre_depths = []
        projector_view_matrices = []
        projector_proj_matrices = []
        for file in texture_files:
            name = file.stem  # Get the filename without extension
            #yaw, pitch = map(float, name.split('_')[-2:])

            if len(name.split('_')) != 4:
                yaw, pitch = map(float, name.split('_')[-2:])
                zoom = 1.
            else:
                yaw, pitch, zoom = map(float, name.split('_')[-3:])

            raw_depth = self.render_depth(180 + pitch, yaw, (width, height), zoom=zoom, raw_return=True)
            pre_depths.append(raw_depth)
            view, proj = self.get_view_projection_matrices(180 + pitch, yaw, zoom=zoom)

            # WARN: Extremely important! Row major to column major
            projector_view_matrices.append(np.array(view).T)
            projector_proj_matrices.append(np.array(proj).T)

        self.projector_view_matrices = np.array(projector_view_matrices, dtype='f4')
        self.projector_proj_matrices = np.array(projector_proj_matrices, dtype='f4')

        # Helper to pack UBO data
        def pack_ubo(directions, views, projs, active, refined):
            data = []
            # 1. Directions (vec4 array)
            dirs_padded = np.array([list(d) + [0.0] for d in directions], dtype='f4')
            data.append(dirs_padded.tobytes())
            # 2. Views (mat4 array)
            data.append(views.tobytes())
            # 3. Projs (mat4 array)
            data.append(projs.tobytes())
            # 4. Active (ivec4 array)
            active_padded = np.zeros((len(active), 4), dtype='int32')
            active_padded[:, 0] = active
            data.append(active_padded.tobytes())
            # 5. Refined (ivec4 array)
            refined_padded = np.zeros((len(refined), 4), dtype='int32')
            refined_padded[:, 0] = refined
            data.append(refined_padded.tobytes())
            return b''.join(data)

        ubo_data = pack_ubo(self.projector_directions, self.projector_view_matrices, self.projector_proj_matrices, self.active_textures, self.refined_textures)
        self.projector_ubo = self.ctx.buffer(ubo_data)
        self.projector_ubo.bind_to_uniform_block(0)  # Bind to binding point 0
        self.bake_prog['Projectors'].binding = 0  # Associate the uniform block with the binding point

        self.depth_maps = self.ctx.texture_array(
                            size=(width, height, texture_count), 
                            components=1, 
                            alignment=4, 
                            data=np.array(pre_depths), 
                            dtype='f4',)

        self.bake_prog['depth_maps'] = 1
        self.depth_maps.use(location=1)


    def render_depth(self, pitch, yaw, img_res, zoom=1., raw_return=False):
        view_matrix, projection_matrix = self.get_view_projection_matrices(pitch, yaw, zoom)

        # Create a floating-point color texture to store depth values
        depth_tex = self.ctx.texture(img_res, components=1, dtype='f4')
        depth_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        depth_tex.repeat_x = False
        depth_tex.repeat_y = False
    
        # Use a depth renderbuffer for depth testing
        depth_rb = self.ctx.depth_renderbuffer(img_res)
    
        fbo = self.ctx.framebuffer(color_attachments=[depth_tex], depth_attachment=depth_rb)
        fbo.use()
        fbo.clear(1.0, 1.0, 1.0, 1.0)
    
        #self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
    
        # Write matrices and uniforms to shader
        self.depth_prog['projection'].write(projection_matrix)
        self.depth_prog['view'].write(view_matrix)
        self.depth_prog['model'].write(glm.mat4(1.0))
    
        # Render the mesh
        self.depth_vao.render()
    
        # Read depth data from color texture
        depth_data = np.frombuffer(depth_tex.read(), dtype='f4')
        depth_image_raw = depth_data.reshape(img_res)
    
        if raw_return:
            return depth_image_raw
        else:
            depth_image_normalized = ((depth_image_raw + 1.0) / 2.0 * 255).astype(np.uint8)  # Map from [-1, 1] to [0, 255]
            depth_image_pil = Image.fromarray(depth_image_normalized, mode='L').transpose(Image.FLIP_TOP_BOTTOM)
            return depth_image_pil

    def bake_texture(self, uv_texture_size):
        # Create the framebuffer
        fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture(uv_texture_size, components=4)],
            depth_attachment=self.ctx.depth_renderbuffer(uv_texture_size)
        )
        fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.disable(moderngl.DEPTH_TEST)
        # Render the mesh
        self.vao.render()

        # Read the result
        data = fbo.read(components=4)
        image = Image.frombytes('RGBA', uv_texture_size, data).transpose(Image.FLIP_TOP_BOTTOM)
        return image

    def calculate_projector_directions(self, texture_files):
        directions = []
        for file in texture_files:
            # Extract pitch and yaw from the filename
            name = file.stem  # Get the filename without extension
            try:
                #yaw, pitch = map(float, name.split('_')[-2:])

                if len(name.split('_')) != 4:
                    yaw, pitch = map(float, name.split('_')[-2:])
                    zoom = 1.
                else:
                    yaw, pitch, zoom = map(float, name.split('_')[-3:])
            except ValueError:
                raise ValueError(f"Texture filename '{name}' does not follow 'pitch_yaw' format.")
            directions.append(self.calculate_camera_position(pitch, yaw))
        return directions

    def fetch_projector_pitch_yaw(self, texture_files):
        pitch_yaw = []
        for file in texture_files:
            # Extract pitch and yaw from the filename
            name = file.stem  # Get the filename without extension
            try:
                #yaw, pitch = map(float, name.split('_')[-2:])

                if len(name.split('_')) != 4:
                    yaw, pitch = map(float, name.split('_')[-2:])
                    zoom = 1.
                else:
                    yaw, pitch, zoom = map(float, name.split('_')[-3:])
            except ValueError:
                raise ValueError(f"Texture filename '{name}' does not follow 'pitch_yaw' format.")
            pitch_yaw.append((pitch, yaw))
        return pitch_yaw

    def calculate_camera_position(self, pitch, yaw):
        pitch = 180 + pitch
        pitch_rad = glm.radians(pitch)
        yaw_rad = glm.radians(yaw)
        camera_x = glm.cos(pitch_rad) * glm.sin(yaw_rad)
        camera_y = -glm.sin(pitch_rad)
        camera_z = glm.cos(pitch_rad) * glm.cos(yaw_rad)
        return glm.vec3(camera_x, camera_y, camera_z)

    def get_view_projection_matrices(self, pitch, yaw, zoom=1.):
        pitch_rad = glm.radians(pitch)
        yaw_rad   = glm.radians(yaw)

        camera_x =  self.camera_distance * glm.cos(pitch_rad) * glm.sin(yaw_rad)
        camera_y = -self.camera_distance * glm.sin(pitch_rad)
        camera_z =  self.camera_distance * glm.cos(pitch_rad) * glm.cos(yaw_rad)

        camera_position = glm.vec3(camera_x, camera_y, camera_z)
        view_matrix = glm.lookAt(camera_position, glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))

        near   = 0.1
        far    = 5.0
        left   = -0.5 * zoom
        right  =  0.5 * zoom
        bottom = -0.5 * zoom
        top    =  0.5 * zoom
        projection_matrix = glm.ortho(left, right, bottom, top, near, far)
        
        return view_matrix, projection_matrix
    
    @staticmethod
    def assign_baked_texture_to_mesh(obj_path, baked_texture_filename):
        # Read the OBJ file
        with open(obj_path, 'r') as f:
            obj_lines = f.readlines()
    
        # Create the MTL filename
        mtl_filename = 'baked_material.mtl'
    
        # Insert or modify the mtllib line
        found_mtllib = False
        for i, line in enumerate(obj_lines):
            if line.startswith('mtllib'):
                obj_lines[i] = f'mtllib {mtl_filename}\n'
                found_mtllib = True
                break
    
        if not found_mtllib:
            # If no mtllib line, add it at the beginning
            obj_lines.insert(0, f'mtllib {mtl_filename}\n')
    
        # Insert the usemtl line before face definitions
        for i, line in enumerate(obj_lines):
            if line.startswith('f '):
                obj_lines.insert(i, 'usemtl BakedMaterial\n')
                break
    
        # Write the modified OBJ file
        output_obj_path = os.path.splitext(obj_path)[0] + '_with_texture.obj'
        with open(output_obj_path, 'w') as f:
            f.writelines(obj_lines)
    
        # Create the MTL file
        mtl_content = f"""# Material Count: 1
    
    newmtl BakedMaterial
    Ka 0.000000 0.000000 0.000000
    Kd 1.000000 1.000000 1.000000
    Ks 0.000000 0.000000 0.000000
    Tr 1.000000
    illum 1
    Ns 0.000000
    map_Kd {os.path.basename(baked_texture_filename)}
    """
        mtl_path = os.path.join(os.path.dirname(obj_path), mtl_filename)
        with open(mtl_path, 'w') as f:
            f.write(mtl_content)
    
        log.info(f"Modified OBJ saved to {output_obj_path}")
        log.info(f"MTL file saved to {mtl_path}")

    @staticmethod
    def unwrap_mesh_with_xatlas(obj_path, save_dir):
        
        import trimesh
        import xatlas
    
        mesh = trimesh.load_mesh(obj_path)
        vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces, mesh.vertex_normals)
    
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare the output filename
        original_filename = os.path.basename(obj_path)
        new_filename = original_filename.replace('.obj', '_unwrapped.obj')
        unwrapped_obj_path = os.path.join(save_dir, new_filename)
    
        xatlas.export(unwrapped_obj_path, mesh.vertices[vmapping], indices, uvs, mesh.vertex_normals[vmapping])
        
        return unwrapped_obj_path
    
    @staticmethod
    def pad_uvs(image, max_passes=100):
        """
        Optimized version of pad_uvs using convolution to fill transparent pixels.
    
        Parameters:
            image_path (str): Path to the input image with transparency.
            output_path (str): Path to save the output image.
            max_passes (int): Maximum number of passes to perform.
    
        Returns:
            None
        """
        # Convert the image to a NumPy array
        data = np.array(image)
        # The last channel is the alpha channel
        alpha_channel = data.shape[2] - 1
    
        # Create input data buffer
        input_data = data.copy()
    
        # Create masks
        transparent_mask = input_data[:, :, alpha_channel] != 255
        opaque_mask = input_data[:, :, alpha_channel] == 255
    
        total_empty_pixels = np.sum(transparent_mask)
    
        if total_empty_pixels == 0:
            result_image = Image.fromarray(data)
            result_image.save(output_path)
            return
    
        # Define the kernel for 8-neighbor connectivity
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=int)
    
        passes = 0
        while True:
            empty_pixels = np.sum(transparent_mask)
            if empty_pixels == 0:
                log.info("All transparent pixels have been filled.")
                break
            passes += 1
    
            # Count the number of opaque neighbors for each pixel
            counts = convolve(opaque_mask.astype(int), kernel, mode='constant', cval=0)
    
            # Avoid division by zero
            counts_safe = counts.copy()
            counts_safe[counts_safe == 0] = 1  # Temporarily replace zeros to avoid division errors
    
            # Prepare mask for pixels to update (transparent pixels with at least one opaque neighbor)
            mask_to_update = (transparent_mask) & (counts > 0)
    
            # For each color channel, compute the weighted sum of neighboring opaque pixels
            for c in range(alpha_channel):
                # Multiply the color channel by the opaque mask to zero out transparent pixels
                color_channel = input_data[:, :, c] * opaque_mask
    
                # Convolve to sum neighboring colors
                neighbor_sum = convolve(color_channel.astype(float), kernel, mode='constant', cval=0)
    
                # Compute the average color for transparent pixels with at least one opaque neighbor
                average_color = neighbor_sum / counts_safe
    
                # Update the transparent pixels where counts > 0
                input_data[:, :, c][mask_to_update] = average_color[mask_to_update].astype(np.uint8)
    
            # Update alpha channel for the pixels we have updated
            input_data[:, :, alpha_channel][mask_to_update] = 255
    
            # Update masks for the next iteration
            opaque_mask = input_data[:, :, alpha_channel] == 255
            transparent_mask = input_data[:, :, alpha_channel] != 255
    
            # Stop if maximum number of passes is reached
            if passes >= max_passes:
                log.info("Reached maximum number of passes.")
                break
    
        # Save the result image
        result_image = Image.fromarray(input_data)
    
        return result_image

    @staticmethod
    def pad_uvs_cupy(image: Image.Image, max_passes: int = 100) -> Image.Image:
        """
        Optimized version of pad_uvs using GPU acceleration with CuPy to fill transparent pixels.

        Parameters:
            image (PIL.Image.Image): Input image with transparency.
            max_passes (int): Maximum number of passes to perform.

        Returns:
            PIL.Image.Image: Result image with filled pixels.
        """
        # Convert the PIL Image to a NumPy array and then to a CuPy array
        data_cpu = np.array(image)
        data = cp.asarray(data_cpu)
        
        # The last channel is assumed to be the alpha channel
        alpha_channel = data.shape[2] - 1

        # Create a copy of the input data on the GPU
        input_data = data.copy()

        # Create masks for transparent and opaque pixels
        transparent_mask = input_data[:, :, alpha_channel] != 255
        opaque_mask = input_data[:, :, alpha_channel] == 255

        # Define the kernel for 8-neighbor connectivity
        kernel = cp.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=cp.int32)

        passes = 0
        while True:
            # Count the number of transparent pixels remaining
            empty_pixels = cp.sum(transparent_mask)
            if empty_pixels == 0:
                log.info("All transparent pixels have been filled.")
                break
            passes += 1

            # Convolve to count the number of opaque neighbors for each pixel
            counts = convolve(opaque_mask.astype(cp.int32), kernel, mode='constant', cval=0)

            # Avoid division by zero by setting zero counts to one temporarily
            counts_safe = counts.copy()
            counts_safe[counts_safe == 0] = 1

            # Create a mask for pixels that are transparent and have at least one opaque neighbor
            mask_to_update = (transparent_mask) & (counts > 0)

            # If no pixels can be updated, exit early
            if not cp.any(mask_to_update):
                print("No more pixels can be filled.")
                break

            # Iterate over each color channel (excluding alpha)
            for c in range(alpha_channel):
                # Zero out transparent pixels in the current color channel
                color_channel = input_data[:, :, c] * opaque_mask

                # Convolve to sum the neighboring colors
                neighbor_sum = convolve(color_channel.astype(cp.float32), kernel, mode='constant', cval=0)

                # Compute the average color for the pixels to update
                average_color = neighbor_sum / counts_safe

                # Update the color channel with the computed average where applicable
                input_data[:, :, c][mask_to_update] = average_color[mask_to_update].astype(cp.uint8)

            # Update the alpha channel to fully opaque for the updated pixels
            input_data[:, :, alpha_channel][mask_to_update] = 255

            # Update the opaque and transparent masks for the next iteration
            opaque_mask = input_data[:, :, alpha_channel] == 255
            transparent_mask = input_data[:, :, alpha_channel] != 255

            # Check if the maximum number of passes has been reached
            if passes >= max_passes:
                log.info("Reached maximum number of passes.")
                break

        # Transfer the processed data back to the CPU
        result_data_cpu = cp.asnumpy(input_data)

        # Convert the NumPy array back to a PIL Image
        result_image = Image.fromarray(result_data_cpu)

        return result_image