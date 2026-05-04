#version 330 core

in vec3 frag_normal;
in vec3 frag_position;

uniform mat4 view;

out vec4 frag_color;

void main() {
    vec3 view_normal = normalize((view * vec4(frag_normal, 0.0)).xyz);
    vec3 normal_color = view_normal * 0.5 + 0.5; // Map from [-1, 1] to [0, 1]
    frag_color = vec4(normal_color, 1.0);
}
