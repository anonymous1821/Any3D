from pathlib import Path
from PIL import Image
from objloader import Obj

import moderngl
import moderngl_window as mglw
import numpy as np
import glm

# Modified shader generation functions
def generate_fragment_shader_code(texture_count):
    return f"""
    #version 330 core

    in vec3 frag_position;
    in vec3 frag_normal;

    out vec4 frag_color;

    uniform sampler2DArray textures; // Array of textures
    uniform sampler2DArray depth_maps; // Array of depth_maps
    uniform int texture_count;
    uniform int active_textures[{texture_count}]; // Array indicating active textures
    uniform int refined_textures[{texture_count}]; // Array indicating refined textures
    layout(std140) uniform Projectors {{
        vec3 projector_directions[{texture_count}]; // Directions of the projectors
    }};
    vec2 project_to_plane(vec3 position, vec3 direction) {{
        // Use a fixed up vector aligned with the global Y-axis
        vec3 up = vec3(0, 1, 0);

        // Calculate right and forward vectors based on up and direction
        vec3 right = normalize(cross(up, direction));
        vec3 forward = normalize(cross(direction, right));

        // Project the position onto the plane
        return vec2(dot(position, right), dot(position, forward));
    }}

    void main() {{
        vec4 blended_color = vec4(0.0);
        float total_weight = 0.0;

        float refined_weight_factor = 10.0;
        float unrefined_weight_factor = 0.0001;

        for (int i = 0; i < texture_count; ++i) {{
            if (active_textures[i] == 0) continue; // Skip inactive textures

            vec3 projection_dir = normalize(projector_directions[i]);

            // Project the fragment position onto the plane defined by the projector direction
            vec2 tex_coords = project_to_plane(frag_position, projection_dir);
            tex_coords = (tex_coords + 0.5); // Convert to texture coordinates for [-0.5, 0.5] range
            tex_coords.y = 1.0 - tex_coords.y; // Flip the y coordinate
            tex_coords = clamp(tex_coords, 0.0, 1.0); // Ensure coordinates are within the texture bounds

            float alignment = max(dot(normalize(frag_normal), projection_dir), 0.0); // Calculate alignment weight
            float weight = alignment * alignment * alignment; // Use alignment as weight

            // Adjust weight based on whether texture is refined
            if (refined_textures[i] == 1) {{
                weight *= refined_weight_factor;
            }} else {{
                weight *= unrefined_weight_factor;
            }}

            vec4 tex_color = texture(textures, vec3(tex_coords, i));

            total_weight += tex_color.a * weight; // Need to use hard mask
            blended_color.rgb += tex_color.rgb * tex_color.a * weight; // Apply alpha blending for RGB
            blended_color.a   += tex_color.a * weight; // Accumulate alpha separately
        }}

        // Normalize the final color by the total weight to avoid overexposure
        if (total_weight > 0.0) {{
            blended_color.rgb /= total_weight; // Divide RGB by accumulated alpha to normalize
            blended_color.a   /= total_weight; // Normalize alpha by total weight
        }}

        blended_color.a = 1.;
        frag_color = blended_color;
    }}
    """

def generate_cosine_fragment_shader_code(texture_count):
    return f"""
    #version 330 core
    in vec3 frag_position;
    in vec3 frag_normal;
    
    out vec4 frag_color;
    
    uniform int texture_count;
    uniform int active_textures[{texture_count}]; // Array indicating active textures
    uniform int refined_textures[{texture_count}]; // Array indicating refined textures
    layout(std140) uniform Projectors {{
        vec3 projector_directions[{texture_count}]; // Directions of the projectors
    }};
    
    void main() {{
        float total_weight = 0.0;
        float weight_sum = 0.0;

        float refined_weight_factor = 10.0;
        float unrefined_weight_factor = 1.0;

        for (int i = 0; i < texture_count; ++i) {{
            if (active_textures[i] == 0) continue; // Skip inactive textures
    
            vec3 projection_dir = normalize(projector_directions[i]);
            float alignment = dot(normalize(frag_normal), projection_dir); // Calculate alignment weight

            float weight = alignment;
            if (refined_textures[i] == 1) {{
                weight *= refined_weight_factor;
            }} else {{
                weight *= unrefined_weight_factor;
            }}

            total_weight += weight;
            weight_sum += abs(weight);
        }}
    
        // Avoid division by zero
        if (weight_sum > 0.0) {{
            total_weight /= weight_sum;
        }}
    
        // Map the total_weight to a [0,1] range for visualization
        float intensity = total_weight * 0.5 + 0.5;
    
        frag_color = vec4(vec3(intensity), 1.0);
    }}
    """

class TextureProjectionExample(mglw.WindowConfig):
    title = "Texture Projection Example"
    window_size = (1024, 1024)
    aspect_ratio = 1.0
    resource_dir = Path('data')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        vertex_shader_path = './shaders/vertex_shader.glsl'
        obj_path           = '...' # Your path
        texture_dir        = '...' # Your path
        output_dir         = Path('...') # Your path
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load the OBJ mesh
        obj = Obj.open(obj_path)

        # Load multiple textures and create a texture array
        texture_files = list(Path(texture_dir).glob('*.png'))
        textures = [Image.open(file).convert('RGBA') for file in texture_files]
        texture_count = len(textures)

        # Create the refined_textures array
        self.refined_textures = np.zeros(texture_count, dtype='int32')
        for idx, file in enumerate(texture_files):
            if file.parts[-1].split(sep='_')[0] == 'refined':
                self.refined_textures[idx] = 1
            else:
                self.refined_textures[idx] = 0

        # Generate the fragment shader code dynamically
        fragment_shader_code = generate_fragment_shader_code(texture_count)
        cosine_fragment_shader_code = generate_cosine_fragment_shader_code(texture_count)

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
            fragment_shader=open('./shaders/normal_fragment_shader.glsl').read()
        )

        self.depth_prog = self.ctx.program(
            vertex_shader=open(vertex_shader_path).read(),
            fragment_shader='''
                #version 330 core
                void main() {
                    // Depth is automatically handled by OpenGL
                }
            '''
        )

        self.camera_rotation = glm.vec2(0.0, 0.0)
        self.camera_distance = 5.0  # Initial distance of the camera from the object
        self.zoom_level = 1.0  # Initial zoom level

        # Toggles for alpha blending and planes visualization
        self.enable_alpha_blending = True
        self.show_planes = True

        # Track which textures are active (default to all active)
        self.active_textures = np.ones(texture_count, dtype='int32')

        # Properly set the active_textures and refined_textures uniform arrays in the shaders
        self.prog['active_textures'].write(self.active_textures.tobytes())
        self.prog['refined_textures'].write(self.refined_textures.tobytes())
        self.cosine_prog['active_textures'].write(self.active_textures.tobytes())
        self.cosine_prog['refined_textures'].write(self.refined_textures.tobytes())

        width, height = textures[0].size
        texture_array_data = np.array([np.array(tex) for tex in textures], dtype='uint8')
        self.texture_array = self.ctx.texture_array((width, height, texture_count), 4, texture_array_data)

        # Bind textures and set uniforms
        self.prog['textures'] = 0
        self.prog['texture_count'].value = texture_count
        self.texture_array.use(location=0)

        self.cosine_prog['texture_count'].value = texture_count

        # Define projector directions dynamically based on texture filenames
        self.projector_directions = self.calculate_projector_directions(texture_files)

        # Create a uniform buffer for projector directions
        projector_directions_data = np.array([list(dir) + [0.0] for dir in self.projector_directions], dtype='f4').flatten()
        self.projector_ubo = self.ctx.buffer(projector_directions_data.tobytes())
        self.projector_ubo.bind_to_uniform_block(0)  # Bind to binding point 0
        self.prog['Projectors'].binding = 0  # Associate the uniform block with the binding point
        self.cosine_prog['Projectors'].binding = 0  # Associate the uniform block with the binding point

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
        
        # Create planes to show the textures
        self.texture_plane_prog = self.ctx.program(
            vertex_shader=open('./shaders/plane_vertex_shader.glsl').read(),
            fragment_shader=open('./shaders/plane_fragment_shader.glsl').read(),
        )
        self.texture_planes = [self.create_plane(i) for i in range(len(texture_files))]

    def calculate_projector_directions(self, texture_files):
        directions = []
        for file in texture_files:
            # Extract pitch and yaw from the filename
            name = file.stem  # Get the filename without extension
            try:
                yaw, pitch = map(float, name.split('_')[1:])
            except ValueError:
                raise ValueError(f"Texture filename '{name}' does not follow 'pitch_yaw' format.")
            directions.append(self.calculate_camera_position(pitch, yaw))
        return directions

    def calculate_camera_position(self, pitch, yaw):
        pitch = 180 + pitch
        pitch_rad = glm.radians(pitch)
        yaw_rad = glm.radians(yaw)
        camera_x = glm.cos(pitch_rad) * glm.sin(yaw_rad)
        camera_y = -glm.sin(pitch_rad)
        camera_z = glm.cos(pitch_rad) * glm.cos(yaw_rad)
        return glm.vec3(camera_x, camera_y, camera_z)

    def get_view_projection_matrices(self):
        # Calculate the camera position based on rotation and distance
        camera_x = self.camera_distance * glm.cos(self.camera_rotation.y) * glm.cos(self.camera_rotation.x)
        camera_y = self.camera_distance * glm.sin(self.camera_rotation.x)
        camera_z = self.camera_distance * glm.sin(self.camera_rotation.y) * glm.cos(self.camera_rotation.x)

        camera_position = glm.vec3(camera_x, camera_y, camera_z)
        view_matrix = glm.lookAt(camera_position, glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))

        # Set up orthographic projection with zoom
        zoom_factor = self.zoom_level
        left        = -self.window_size[0] / 200.0 * zoom_factor
        right       = self.window_size[0] / 200.0 * zoom_factor
        bottom      = -self.window_size[1] / 200.0 * zoom_factor
        top         = self.window_size[1] / 200.0 * zoom_factor
        near        = 0.1
        far         = 1000.0
        projection_matrix = glm.ortho(left, right, bottom, top, near, far)
        
        return view_matrix, projection_matrix

    def create_plane(self, layer):
        size = 0.5  # Size of the planes to match [-0.5, 0.5] range
        vertices = np.array([
            -size, -size, 0.0, 0.0, 1.0,  # Note: flipped the y texture coordinates
             size, -size, 0.0, 1.0, 1.0,
             size,  size, 0.0, 1.0, 0.0,
            -size,  size, 0.0, 0.0, 0.0,
        ], dtype='f4')

        indices = np.array([
            0, 1, 2,
            2, 3, 0
        ], dtype='i4')

        vbo = self.ctx.buffer(vertices.tobytes())
        ibo = self.ctx.buffer(indices.tobytes())
        vao = self.ctx.vertex_array(self.texture_plane_prog, [
            (vbo, '3f 2f', 'in_position', 'in_texcoord')
        ], index_buffer=ibo)
        vao.layer = layer  # Store the texture layer to sample
        return vao

    def render(self, time: float, frame_time: float):
        view_matrix, projection_matrix = self.get_view_projection_matrices()

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

        # Render texture planes if the toggle is enabled
        if self.show_planes:
            for i, plane in enumerate(self.texture_planes):
                model_matrix = glm.mat4(1.0)
                model_matrix = glm.translate(model_matrix, self.projector_directions[i])
                rot_matrix   = glm.lookAt(self.projector_directions[i], glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
                model_matrix = model_matrix * glm.transpose(rot_matrix)
                self.texture_plane_prog['model'].write(model_matrix)
                self.texture_plane_prog['projection'].write(projection_matrix)
                self.texture_plane_prog['view'].write(view_matrix)
                self.texture_array.use(location=0)
                self.texture_plane_prog['layer'].value = plane.layer
                plane.render()

    def mouse_drag_event(self, x, y, dx, dy):
        self.camera_rotation.y += dx * 0.01
        self.camera_rotation.x += dy * 0.01

    def mouse_scroll_event(self, x_offset, y_offset):
        self.zoom_level -= y_offset * 0.03
        self.zoom_level = max(0.03, self.zoom_level)  # Prevent zooming too close

    def key_event(self, key, action, modifiers):
        # Toggle alpha blending with 'B'
        if key == self.wnd.keys.B and action == self.wnd.keys.ACTION_PRESS:
            self.enable_alpha_blending = not self.enable_alpha_blending
            print(f"Alpha Blending {'Enabled' if self.enable_alpha_blending else 'Disabled'}")

        # Toggle planes visualization with 'P'
        if key == self.wnd.keys.P and action == self.wnd.keys.ACTION_PRESS:
            self.show_planes = not self.show_planes
            print(f"Planes Visualization {'Enabled' if self.show_planes else 'Disabled'}")

        # Toggle specific texture projection with number keys 1-6
        if key in [self.wnd.keys.NUMBER_1, self.wnd.keys.NUMBER_2, self.wnd.keys.NUMBER_3, self.wnd.keys.NUMBER_4, self.wnd.keys.NUMBER_5, self.wnd.keys.NUMBER_6] and action == self.wnd.keys.ACTION_PRESS:
            index = key - self.wnd.keys.NUMBER_1
            self.active_textures[index] = not self.active_textures[index]
            self.prog['active_textures'].write(self.active_textures.tobytes())
            self.cosine_prog['active_textures'].write(self.active_textures.tobytes())
            print(f"Texture {index + 1} {'Enabled' if self.active_textures[index] else 'Disabled'}")

if __name__ == '__main__':
    mglw.run_window_config(TextureProjectionExample)


