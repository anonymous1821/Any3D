#version 430 core

// Define the number of work items in each workgroup
layout(local_size_x = 256) in;

// Buffer bindings
layout(std430, binding = 0) buffer Vertices {
    vec3 vertices[];
};

layout(std430, binding = 1) buffer Faces {
    ivec3 faces[];
};

layout(std430, binding = 2) buffer FaceFlags {
    int face_flags[];
};

// Uniforms
uniform mat4 mvp_matrix;
uniform ivec2 image_size;

// Texture binding (use a different binding point to avoid confusion)
layout(binding = 3) uniform usampler2D silhouette_texture;

// Image binding for output (binary projection result)
layout(binding = 4, rgba8) uniform writeonly image2D projection_output;

// Function to compute barycentric coordinates
vec3 barycentric_coords(vec2 a, vec2 b, vec2 c, vec2 p) {
    vec2 v0 = b - a, v1 = c - a, v2 = p - a;
    float d00 = dot(v0, v0), d01 = dot(v0, v1), d11 = dot(v1, v1);
    float d20 = dot(v2, v0), d21 = dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    if (denom == 0.0) return vec3(-1.0); // Degenerate triangle
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0 - v - w;
    return vec3(u, v, w);
}

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
    uint face_idx = gl_GlobalInvocationID.x;

    ivec3 face = faces[face_idx];

    vec4 v0 = vec4(vertices[face.x], 1.0) * mvp_matrix;
    vec4 v1 = vec4(vertices[face.y], 1.0) * mvp_matrix;
    vec4 v2 = vec4(vertices[face.z], 1.0) * mvp_matrix;

    v0 /= v0.w;
    v1 /= v1.w;
    v2 /= v2.w;

    // Corrected conversion to pixel coordinates with Y-axis flip
    vec2 p0 = vec2(
        (v0.x * 0.5 + 0.5) * image_size.x,
        (1.0 - (v0.y * 0.5 + 0.5)) * image_size.y
    );
    vec2 p1 = vec2(
        (v1.x * 0.5 + 0.5) * image_size.x,
        (1.0 - (v1.y * 0.5 + 0.5)) * image_size.y
    );
    vec2 p2 = vec2(
        (v2.x * 0.5 + 0.5) * image_size.x,
        (1.0 - (v2.y * 0.5 + 0.5)) * image_size.y
    );

    // Compute bounding box
    ivec2 min_coords = ivec2(floor(min(min(p0, p1), p2)));
    ivec2 max_coords = ivec2(ceil(max(max(p0, p1), p2)));

    // Clamp to image bounds
    min_coords = clamp(min_coords, ivec2(0), image_size - 1);
    max_coords = clamp(max_coords, ivec2(0), image_size - 1);

    bool overlap = true;

    // Rasterize the triangle
    for (int y = min_coords.y; y <= max_coords.y; y++) {
        for (int x = min_coords.x; x <= max_coords.x; x++) {
            vec2 p = vec2(x, y) + vec2(0.5);

            // Compute barycentric coordinates
            vec3 bary = barycentric_coords(p0.xy, p1.xy, p2.xy, p);

            if (bary.x >= 0 && bary.y >= 0 && bary.z >= 0) {
                // Sample the silhouette mask
                uint mask_value = texelFetch(silhouette_texture, ivec2(x, y), 0).r;

                if (mask_value == 0) {
                    overlap = false;
                    imageStore(projection_output, ivec2(x, y), vec4(1.0));
                    break;
                }
                else {
                    // overlap = true;
                    imageStore(projection_output, ivec2(x, y), vec4(0., 0., 0., 1.));
                    // break;
                }
            }
        }
    }

    face_flags[face_idx] = overlap ? 1 : 0;
}
