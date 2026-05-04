#version 330 core

in vec3 in_vert;
in vec3 in_norm;

out vec3 frag_position;
out vec3 frag_normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    vec4 world_position = model * vec4(in_vert, 1.0);
    frag_position = world_position.xyz;
    frag_normal = mat3(transpose(inverse(model))) * in_norm;
    gl_Position = projection * view * world_position;
}

