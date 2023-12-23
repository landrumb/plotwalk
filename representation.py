import math
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import euclidean

def interpolate_line_segment(start, end, num_points):
    return np.linspace(start, end, num_points)

def gaussian_weight(distance, variance):
    return norm.pdf(distance, 0, variance**0.5)

def perpendicular_distance_and_projection(point, line_start, line_end):
    line_vec = np.array(line_end) - np.array(line_start)
    point_vec = np.array(point) - np.array(line_start)
    
    line_vec_normalized = line_vec / np.linalg.norm(line_vec)
    projection_length = np.dot(point_vec, line_vec_normalized)
    
    # Check if the projection falls within the line segment
    if projection_length < 0 or projection_length > np.linalg.norm(line_vec):
        return None  # Projection is outside the line segment

    nearest_point = np.array(line_start) + projection_length * line_vec_normalized
    return euclidean(point, nearest_point)

def path_integral(matrix, start_point, end_point, line_start, line_end, variance):
    num_points = int(euclidean(start_point, end_point)) * 2
    line_points = interpolate_line_segment(np.array(start_point), np.array(end_point), num_points)

    integral = 0
    for point in line_points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < matrix.shape[1] and 0 <= y < matrix.shape[0]:
            distance_to_line = perpendicular_distance_and_projection(point, line_start, line_end)
            if distance_to_line is not None:
                darkness = matrix[y, x]
                weight = gaussian_weight(distance_to_line, variance)
                integral += darkness * weight

    return integral


def point_on_inscribed_circle(m, n, theta):
    # Radius is half the smaller of the two dimensions
    radius = min(m, n) / 2

    # Center of the rectangle
    center_x = m / 2
    center_y = n / 2

    # Calculate the coordinates
    x = center_x + radius * math.cos(theta)
    y = center_y + radius * math.sin(theta)

    return x, y

