import glm


def calculate_camera_position(pitch, yaw):
    pitch = 180 + pitch
    pitch_rad = glm.radians(pitch)
    yaw_rad   = glm.radians(yaw)
    camera_x  = glm.cos(pitch_rad) * glm.sin(yaw_rad)
    camera_y  = -glm.sin(pitch_rad)
    camera_z  = glm.cos(pitch_rad) * glm.cos(yaw_rad)
    return glm.vec3(camera_x, camera_y, camera_z)

def get_view_projection_matrices(camera_distance, pitch, yaw, zoom=1.):
    pitch_rad = glm.radians(pitch)
    yaw_rad   = glm.radians(yaw)

    camera_x =  camera_distance * glm.cos(pitch_rad) * glm.sin(yaw_rad)
    camera_y = -camera_distance * glm.sin(pitch_rad)
    camera_z =  camera_distance * glm.cos(pitch_rad) * glm.cos(yaw_rad)

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
