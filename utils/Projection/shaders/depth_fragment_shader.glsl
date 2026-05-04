#version 330 core
in vec3 frag_position_cam;
out vec4 FragColor;

uniform float near = 0.1; 
uniform float far = 1000.0;

float LinearizeDepth(float depth) {
    float z = depth * 2.0 - 1.0; // Back to NDC
    return (2.0 * near * far) / (far + near - z * (far - near));	
}

void main() {
    float depth = LinearizeDepth(gl_FragCoord.z) / far; // Normalize depth
    FragColor = vec4(vec3(depth), 1.0);
}
