#version 330 core

in vec3 frag_position;
in vec3 frag_normal;

out vec4 frag_color;

uniform sampler2DArray textures; // Array of textures
uniform int texture_count;
uniform int active_textures[6]; // Increased size to handle more textures
layout(std140) uniform Projectors {
    vec3 projector_directions[6]; // Increased size to handle more projectors
};
uniform float blend_sharpness = 1.0; // Controls the sharpness of the blending

vec2 project_to_plane(vec3 position, vec3 direction) {
    // Use a fixed up vector aligned with the global Y-axis
    vec3 up = vec3(0, 1, 0);
    
    // Calculate right and forward vectors based on up and direction
    vec3 right = normalize(cross(up, direction));
    vec3 forward = normalize(cross(direction, right));
    
    // Project the position onto the plane
    return vec2(dot(position, right), dot(position, forward));
}

void main() {
    vec4 blended_color = vec4(0.0);
    float total_weight = 0.0;

    for (int i = 0; i < texture_count; ++i) {
        if (active_textures[i] == 0) continue; // Skip inactive textures

        vec3 projection_dir = normalize(projector_directions[i]);
        float alignment = max(dot(normalize(frag_normal), projection_dir), 0.0); // Calculate alignment weight
        float weight = alignment; // Use alignment as weight
        total_weight += weight;

        // Project the fragment position onto the plane defined by the projector direction
        vec2 tex_coords = project_to_plane(frag_position, projection_dir);
        tex_coords = (tex_coords + 0.5); // Convert to texture coordinates for [-0.5, 0.5] range
        tex_coords.y = 1.0 - tex_coords.y; // Flip the y coordinate
        tex_coords = clamp(tex_coords, 0.0, 1.0); // Ensure coordinates are within the texture bounds

        vec4 tex_color = texture(textures, vec3(tex_coords, i));
        blended_color += tex_color * weight;
    }

    // Normalize the final color by the total weight to avoid overexposure
    if (total_weight > 0.0) {
        blended_color /= total_weight;
    }
    frag_color = blended_color;
}
