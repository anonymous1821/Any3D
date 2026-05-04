#version 330 core

in vec3 in_vert;
in vec3 in_norm;
in vec2 in_uv;  // UV coordinates

out vec3 frag_position;
out vec3 frag_normal;
out vec2 frag_uv;

void main() {
    frag_position = in_vert;
    frag_normal = in_norm;
    frag_uv = in_uv;

    // Map UV coordinates from [0, 1] to [-1, 1] for NDC space
    gl_Position = vec4(frag_uv * 2.0 - 1.0, 0.0, 1.0);
}
