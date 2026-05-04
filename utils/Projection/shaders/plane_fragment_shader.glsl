#version 330 core

in vec2 v_texcoord;
out vec4 out_color;

uniform sampler2DArray textures;
uniform int layer;

void main() {
    vec4 tex_color = texture(textures, vec3(v_texcoord, layer));
    tex_color.a = 0.5; // Set alpha to 0.5 for transparency
    out_color = tex_color;
}

