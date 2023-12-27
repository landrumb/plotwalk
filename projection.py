import numpy as np
import torch

def project_point_onto_line(point, line_point1, line_point2):
    A = np.array(line_point1)
    B = np.array(line_point2)
    P = np.array(point)
    AB = B - A
    AP = P - A
    t = np.dot(AP, AB) / np.dot(AB, AB)
    return A + t * AB

def relative_placement(point, line_point1, line_point2):
    """Assuming the point is on the line, point1 is 0 and point2 is 1, what is the relative placement of the point?"""
    A = np.array(line_point1)
    B = np.array(line_point2)
    P = np.array(point)
    AB = B - A
    AP = P - A
    t = np.dot(AP, AB) / np.dot(AB, AB)
    return t

def projection_placement_distance(point, line_point1, line_point2):
    """tuple of (projection, relative_placement, distance)"""
    A = np.array(line_point1)
    B = np.array(line_point2)
    P = np.array(point)
    AB = B - A
    AP = P - A
    t = np.dot(AP, AB) / np.dot(AB, AB)
    projection = A + t * AB
    relative_placement = t
    distance = np.linalg.norm(projection - P)
    return projection, relative_placement, distance

# Switching to pytorch for differentiability
def projection_placement_distance_torch(point, line_point1, line_point2):
    """tuple of (projection, relative_placement, distance) using PyTorch"""
    A = torch.tensor(line_point1, dtype=torch.float32)
    B = torch.tensor(line_point2, dtype=torch.float32)
    P = torch.tensor(point, dtype=torch.float32)

    AB = B - A
    AP = P - A
    t = torch.dot(AP, AB) / torch.dot(AB, AB)
    projection = A + t * AB
    relative_placement = t  # Keep as tensor for autograd compatibility
    distance = torch.norm(projection - P)  # Keep as tensor for autograd compatibility

    return projection, relative_placement, distance



if __name__ == '__main__':
    line_point1 = [1, 2]
    line_point2 = [4, 6]
    point = [0, 0]

    projected_point = project_point_onto_line(point, line_point1, line_point2)
    print(projected_point)

    print(relative_placement(projected_point, line_point1, line_point2))
