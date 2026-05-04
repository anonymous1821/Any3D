from pathlib import Path

from PIL import Image
from objloader import Obj

import moderngl
import numpy as np
import glm

from .mesh.mesh import Mesh, safe_normalize
from .shaders.generate_shader import generate_fragment_shader_code, generate_cosine_fragment_shader_code

import logging


def recompute_normals(vertices, faces):
    normals = np.zeros(vertices.shape, dtype=vertices.dtype)
    tris = vertices[faces]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n = n / np.linalg.norm(n, axis=1)[:, np.newaxis]
    for i in range(3):
        normals[faces[:, i]] += n
    normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
    return normals


class HeadlessProjectionMapping:
    def __init__(self, 
                 vertex_shader_path,
                 normal_fragment_shader_path,
                 obj_mesh,
                 texture_dir,
                 device_idx,
                 ):

        # Configure logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Context creation
        self.ctx = moderngl.create_context(standalone=True, backend='egl', device_index=device_idx)
        self.ctx.gc_mode = 'auto'

        # self.logger.info("Created standalone OpenGL context.")

        # Load the OBJ mesh
        obj = obj_mesh #Obj.open(obj_path)
        # self.logger.info(f"Loaded OBJ mesh from {obj_path}.")

        # Load multiple textures and create a texture array
        texture_files = sorted(Path(texture_dir).glob('*.png'))
        if not texture_files:
            self.logger.error(f"No texture files found in {texture_dir}.")
            raise FileNotFoundError(f"No texture files found in {texture_dir}.")
        
        textures = [Image.open(file).convert('RGBA') for file in texture_files]
        texture_count = len(textures)
        # self.logger.info(f"Loaded {texture_count} textures from {texture_dir}.")

        # self.refined_textures = np.array([
        #     1 if file.stem.startswith('refined') else 0 for file in texture_files
        # ], dtype='int32')

        # Generating the refined_textures array
        self.refined_textures = np.array([
            1 if file.stem.startswith(('refined', 'to_be_refined_')) else 0 
            for file in texture_files
        ], dtype='int32')

        self.actual_refined_textures = np.array([
            1 if file.stem.startswith('refined') else 0 
            for file in texture_files
        ], dtype='int32')

        # Generate the fragment shader code dynamically
        fragment_shader_code = generate_fragment_shader_code(texture_count)
        cosine_fragment_shader_code = generate_cosine_fragment_shader_code(texture_count)
        # self.logger.info("Generated dynamic fragment shader codes.")

        # Compile the shaders
        self.prog = self.ctx.program(
            vertex_shader=open(vertex_shader_path).read(),
            fragment_shader=fragment_shader_code
        )
        self.cosine_prog = self.ctx.program(
            vertex_shader=open(vertex_shader_path).read(),
            fragment_shader=cosine_fragment_shader_code
        )
        self.normal_prog = self.ctx.program(
            vertex_shader=open(vertex_shader_path).read(),
            fragment_shader=open(normal_fragment_shader_path).read()
        )
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
        self.cam_cos_prog = self.ctx.program(
            vertex_shader=open(vertex_shader_path).read(),
            fragment_shader="""
            #version 330 core
            in vec3 frag_position;
            in vec3 frag_normal;

            out vec4 frag_color;

            uniform vec3 camera_position;

            void main()
            {
                vec3 view_dir   = normalize(frag_position - camera_position);
                float alignment = clamp(dot(normalize(frag_normal), view_dir), 0.0, 1.0);
                frag_color = vec4(vec3(alignment), 1.0);
            }
            """
        )

        # self.logger.info("Compiled all shaders successfully.")

        self.camera_distance = 1.5  # Initial distance of the camera from the object
        self.epsilon = 0.001

        # Toggles for alpha blending
        self.enable_alpha_blending = False

        # Track which textures are active (default to all active)
        self.active_textures = np.ones(texture_count, dtype='int32')

        self.prog['epsilon'].value = self.epsilon
        self.cosine_prog['epsilon'].value = self.epsilon

        # Create texture array
        width, height = textures[0].size
        texture_array_data = np.array([np.array(tex) for tex in textures], dtype='uint8')
        self.texture_array = self.ctx.texture_array((width, height, texture_count), 4, texture_array_data.tobytes())
        self.texture_array.use(location=0)
        # self.logger.info("Created and bound texture array.")

        # Set uniforms
        self.prog['textures'] = 0
        # self.prog['texture_count'].value = texture_count

        self.cosine_prog['textures'] = 0
        # self.cosine_prog['texture_count'].value = texture_count

        # Define projector directions dynamically based on texture filenames
        self.projector_directions = self.calculate_projector_directions(texture_files)
        # self.logger.info("Calculated projector directions.")
        
        # Create buffers and VAO for the mesh
        self.vbo = self.ctx.buffer(obj.pack('vx vy vz nx ny nz'))
        self.vbo_depth = self.ctx.buffer(obj.pack('vx vy vz'))
        self.vao = self.ctx.vertex_array(self.prog, [
            (self.vbo, '3f 3f', 'in_vert', 'in_norm')
        ])
        self.normal_vao = self.ctx.vertex_array(self.normal_prog, [
            (self.vbo, '3f 3f', 'in_vert', 'in_norm')
        ])
        self.depth_vao = self.ctx.vertex_array(self.depth_prog, [
            (self.vbo_depth, '3f', 'in_vert')
        ])
        self.cosine_vao = self.ctx.vertex_array(self.cosine_prog, [
            (self.vbo, '3f 3f', 'in_vert', 'in_norm')
        ])
        self.cam_cos_vao = self.ctx.vertex_array(self.cam_cos_prog, [
            (self.vbo, '3f 3f', 'in_vert', 'in_norm')
        ])

        # self.logger.info("Created VAOs for all shader programs.")

        pre_depths = []
        projector_view_matrices = []
        projector_proj_matrices = []
        for file in texture_files:
            name = file.stem  # Get the filename without extension
            # yaw, pitch = map(float, name.split('_')[-2:])
            # raw_depth = self.render_depth(180 + pitch, yaw, (width, height), raw_return=True)
            # pre_depths.append(raw_depth)
            # view, proj = self.get_view_projection_matrices(180 + pitch, yaw, zoom=1.)
            
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

        self.depth_maps = self.ctx.texture_array(
                            size=(width, height, texture_count), 
                            components=1, 
                            alignment=4, 
                            data=np.array(pre_depths), 
                            dtype='f4',)

        self.prog['depth_maps'] = 1
        self.depth_maps.use(location=1)
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

        # Create UBO for main program
        ubo_data = pack_ubo(self.projector_directions, self.projector_view_matrices, self.projector_proj_matrices, self.active_textures, self.refined_textures)
        self.projector_ubo = self.ctx.buffer(ubo_data)
        self.projector_ubo.bind_to_uniform_block(0)
        self.prog['Projectors'].binding = 0

        # Create UBO for cosine program
        cosine_ubo_data = pack_ubo(self.projector_directions, self.projector_view_matrices, self.projector_proj_matrices, self.active_textures, self.actual_refined_textures)
        self.cosine_projector_ubo = self.ctx.buffer(cosine_ubo_data)
        self.cosine_projector_ubo.bind_to_uniform_block(1)
        self.cosine_prog['Projectors'].binding = 1

        self.cosine_prog['depth_maps'] = 1

    def calculate_projector_directions(self, texture_files):
        directions = []
        for file in texture_files:
            # Extract pitch and yaw from the filename
            name = file.stem  # Get the filename without extension
            try:
                # Assuming filenames contain pitch and yaw separated by '_', e.g., 'refined_180_0'
                # parts = name.split('_')
                # yaw = float(parts[-3])
                # pitch = float(parts[-2])
                # zoom = float(parts[-1])

                if len(name.split('_')) != 4:
                    yaw, pitch = map(float, name.split('_')[-2:])
                    zoom = 1.
                else:
                    yaw, pitch, zoom = map(float, name.split('_')[-3:])

            except (ValueError, IndexError):
                self.logger.error(f"Texture filename '{name}' does not follow 'pitch_yaw' format.")
                raise ValueError(f"Texture filename '{name}' does not follow 'pitch_yaw' format.")
            directions.append(self.calculate_camera_position(pitch, yaw))
        return directions

    def calculate_camera_position(self, pitch, yaw):
        pitch += 180
        pitch_rad = glm.radians(pitch)
        yaw_rad = glm.radians(yaw)
        camera_x = glm.cos(pitch_rad) * glm.sin(yaw_rad)
        camera_y = -glm.sin(pitch_rad)
        camera_z = glm.cos(pitch_rad) * glm.cos(yaw_rad)
        return glm.vec3(camera_x, camera_y, camera_z)

    def get_view_projection_matrices(self, pitch, yaw, zoom=1.0):
        pitch_rad = glm.radians(pitch)
        yaw_rad = glm.radians(yaw)

        camera_x = self.camera_distance * glm.cos(pitch_rad) * glm.sin(yaw_rad)
        camera_y = -self.camera_distance * glm.sin(pitch_rad)
        camera_z = self.camera_distance * glm.cos(pitch_rad) * glm.cos(yaw_rad)

        camera_position = glm.vec3(camera_x, camera_y, camera_z)
        view_matrix = glm.lookAt(camera_position, glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))

        # Orthographic projection
        near = 0.1
        far = 5.0
        left = -0.5 * zoom
        right = 0.5 * zoom
        bottom = -0.5 * zoom
        top = 0.5 * zoom
        projection_matrix = glm.ortho(left, right, bottom, top, near, far)

        return view_matrix, projection_matrix

    def render(self, pitch, yaw, img_res, zoom=1.0):
        view_matrix, projection_matrix = self.get_view_projection_matrices(pitch, yaw, zoom)

        # Create an off-screen framebuffer
        fbo = self.ctx.framebuffer(
            color_attachments=self.ctx.texture(img_res, 4),
            depth_attachment=self.ctx.depth_renderbuffer(img_res)
        )
        fbo.use()
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        # Enable or disable alpha blending based on the toggle
        if self.enable_alpha_blending:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        else:
            self.ctx.disable(moderngl.BLEND)

        # Write matrices to shader
        self.prog['projection'].write(projection_matrix)
        self.prog['view'].write(view_matrix)
        self.prog['model'].write(glm.mat4(1.0))  # No rotation of the object itself

        # Render the mesh
        self.vao.render()

        # Read the framebuffer content into a numpy array
        data = fbo.read(components=4)
        image = Image.frombytes('RGBA', img_res, data).transpose(Image.FLIP_TOP_BOTTOM)
        return image

    def render_normal(self, pitch, yaw, img_res, zoom=1.0):
        view_matrix, projection_matrix = self.get_view_projection_matrices(pitch, yaw, zoom)

        # Create an off-screen framebuffer
        fbo = self.ctx.framebuffer(
            color_attachments=self.ctx.texture(img_res, 4),
            depth_attachment=self.ctx.depth_renderbuffer(img_res)
        )
        fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        # Write matrices to shader
        self.normal_prog['projection'].write(projection_matrix)
        self.normal_prog['view'].write(view_matrix)
        self.normal_prog['model'].write(glm.mat4(1.0))  # No rotation of the object itself

        # Render the mesh
        self.normal_vao.render()

        # Read the framebuffer content into a numpy array
        data = fbo.read(components=4)
        image = Image.frombytes('RGBA', img_res, data).transpose(Image.FLIP_TOP_BOTTOM)
        return image

    def render_depth(self, pitch, yaw, img_res, zoom=1., raw_return=False):
        view_matrix, projection_matrix = self.get_view_projection_matrices(pitch, yaw, zoom)

        depth_tex = self.ctx.texture(img_res, components=1, dtype='f4')
        depth_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        depth_tex.repeat_x = False
        depth_tex.repeat_y = False
    
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
        self.logger.debug("Rendered mesh with depth shader.")

        depth_data = np.frombuffer(depth_tex.read(), dtype='f4')
        depth_image_raw = depth_data.reshape(img_res)
        
        disp = 1 / depth_image_raw
        disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-20)

        if raw_return:
            result = depth_image_raw
        else:
            disp[depth_image_raw == 1.] = 0.
            disp_image_normalized = (disp * 255).astype(np.uint8)
            result = Image.fromarray(disp_image_normalized, mode='L').transpose(Image.FLIP_TOP_BOTTOM)

        fbo.release()
        depth_tex.release()
        depth_rb.release()

        return result


    def render_cosine(self, pitch, yaw, img_res, zoom=1.0):
        view_matrix, projection_matrix = self.get_view_projection_matrices(pitch, yaw, zoom)

        # Create an off-screen framebuffer
        fbo = self.ctx.framebuffer(
            color_attachments=self.ctx.texture(img_res, 4),
            depth_attachment=self.ctx.depth_renderbuffer(img_res)
        )
        fbo.use()
        self.ctx.clear(1.0, 1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        # Write matrices to shader
        self.cosine_prog['projection'].write(projection_matrix)
        self.cosine_prog['view'].write(view_matrix)
        self.cosine_prog['model'].write(glm.mat4(1.0))  # No rotation of the object itself

        # Render the mesh
        self.cosine_vao.render()
        self.logger.debug("Rendered mesh with cosine similarity shader.")

        # Read the framebuffer content into a numpy array
        data = fbo.read(components=4)
        image = Image.frombytes('RGBA', img_res, data).transpose(Image.FLIP_TOP_BOTTOM)

        # Convert RGBA image to grayscale cosine similarity map
        cosine_map = np.array(image)[:, :, 0] / 255.0

        # Convert normalized cosine similarity map to an image
        cosine_image = Image.fromarray((cosine_map * 255).astype(np.uint8))
        self.logger.debug("Captured cosine similarity image from framebuffer.")
        return cosine_image

    def render_cam_cos(self, pitch, yaw, img_res, zoom=1.0):
        # Compute camera transform
        view_matrix, projection_matrix = self.get_view_projection_matrices(pitch, yaw, zoom)
        cam_pos = self.calculate_camera_position(pitch, yaw)

        # Create an off-screen framebuffer
        fbo = self.ctx.framebuffer(
            color_attachments=self.ctx.texture(img_res, 4),
            depth_attachment=self.ctx.depth_renderbuffer(img_res)
        )
        fbo.use()
        self.ctx.clear(1.0, 1.0, 1.0, 1.0)
        self.ctx.enable(self.ctx.DEPTH_TEST)

        # Pass matrices to the shader
        self.cam_cos_prog['projection'].write(projection_matrix)
        self.cam_cos_prog['view'].write(view_matrix)
        self.cam_cos_prog['model'].write(glm.mat4(1.0))

        # Pass the camera position to the shader
        self.cam_cos_prog['camera_position'].value = tuple(cam_pos)

        # Render the mesh
        self.cam_cos_vao.render()

        # Read the color buffer from the FBO
        data = fbo.read(components=4)
        image = Image.frombytes('RGBA', img_res, data).transpose(Image.FLIP_TOP_BOTTOM)
        np_image = np.array(image)

        # Extract the red channel (where we put our alignment valcam_pos
        cosine_map = np_image[:, :, 0].astype(np.float32) / 255.0

        # Convert to a PIL Image for convenience
        cosine_image = Image.fromarray((cosine_map * 255).astype(np.uint8))

        return cosine_image


    def calculate_low_cosine_similarity_mask(self, cosine_image, threshold):
        """
        Calculate a mask with pixels that have low cosine similarity values.

        Args:
            cosine_image (PIL.Image): The cosine similarity image.
            threshold (float): The threshold value for low cosine similarity.

        Returns:
            PIL.Image: A binary mask with 255 for low similarity pixels and 0 for others.
        """
        # Convert the cosine similarity image to a NumPy array and normalize to range [0, 1]
        cosine_array = np.array(cosine_image) / 255.0

        # Create a binary mask where pixels below the threshold are set to 255 (low similarity),
        # and pixels above the threshold are set to 0.
        low_similarity_mask = (cosine_array < threshold).astype(np.uint8) * 255

        mask_image = Image.fromarray(low_similarity_mask)
        self.logger.debug(f"Created low cosine similarity mask with threshold {threshold}.")
        return mask_image

    def find_bounding_box(self, rgba_img):
        # Assuming mask is a NumPy array with shape (h, w, c)
        alpha = rgba_img[:, :, -1] > 0  # Extract alpha channel
        if not alpha.any():
            return None

        rows = np.any(alpha, axis=1)
        cols = np.any(alpha, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return (x_min, y_min, x_max, y_max)

    def compute_scale_factor(self, bounding_box, image_width, desired_ratio=0.9):
        if bounding_box is None:
            return 1.0  # Default scale if no object is detected

        image_height = image_width  # Assuming square image; adjust if necessary

        image_center_x = image_width / 2
        image_center_y = image_height / 2

        x_min, y_min, x_max, y_max = bounding_box
        bbox_width  = (x_max - x_min) - min((image_center_x - x_min), (x_max - image_center_x))
        bbox_height = (y_max - y_min) - min((image_center_y - y_min), (y_max - image_center_y))
        current_longer_side = max(bbox_width, bbox_height)

        desired_size = desired_ratio * image_width / 2
        scale_factor = current_longer_side / desired_size

        return scale_factor


    def adjust_camera_zoom(self, mask, image_width, desired_ratio=0.9):
        """
        Adjusts the camera's orthographic projection based on the mask's bounding box.

        Parameters:
            mask (numpy.ndarray): The mask array with shape (h, w).
            image_width (int): The width of the square image.
            desired_ratio (float): Desired ratio of bounding box's longer side to image width.

        Returns:
            float: Scale factor for zoom.
        """
        bounding_box = self.find_bounding_box(mask)
        scale_factor = self.compute_scale_factor(bounding_box, image_width, desired_ratio)

        # self.logger.info(f"Calculated scale factor: {scale_factor} based on bounding box: {bounding_box}")
        return scale_factor