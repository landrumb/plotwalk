# %%
"""Utils for computing a differentiable loss function in PyTorch."""
import torch
from projection import projection_placement_distance_torch, gaussian_weight_torch, doublebatch_projection_placement_distance_torch
from functools import lru_cache
from tqdm import tqdm
import matplotlib.pyplot as plt

def points_on_inscribed_circle(n, width):
    thetas = torch.arange(0, n) * 2 * torch.pi / n
    x = width / 2 + width / 2 * torch.cos(thetas)
    y = width / 2 + width / 2 * torch.sin(thetas)
    points = torch.stack([x, y], dim=1)  # Stacking into a single tensor
    return points


def upper_triangular_coordinate_matrix(n):
    # Create a meshgrid of indices
    rows, cols = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij')

    # Create a mask for the upper triangular part (excluding the diagonal)
    mask = torch.triu(torch.ones(n, n), diagonal=1).bool()

    # Stack the row and column indices into coordinate pairs
    coordinate_pairs = torch.stack((rows[mask], cols[mask]), dim=1)

    return coordinate_pairs

# def add_line_occlusion(occlusion, magnitude, start_point, end_point, variance=0.05):
#     """Adds a line occlusion to the given tensor and returns the result."""
#     dims = torch.tensor(occlusion.shape)
#     for i in range(dims[0]):
#         for j in range(dims[1]):
#             xy = torch.tensor([i / dims[0], j / dims[1]])

#             projection, loc, distance = projection_placement_distance_torch(xy, start_point, end_point)

#             if loc < 0 or loc > 1:
#                 continue
            
#             weight = gaussian_weight_torch(distance, variance)
#             occlusion[i, j] += magnitude * weight

#     return occlusion

def add_line_occlusion(occlusion, magnitude, start_point, end_point, variance=0.05):
    # Create a grid of xy coordinates
    height, width = occlusion.shape
    y_grid, x_grid = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing='ij')
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    # Compute projection, placement, and distance for the entire grid
    _, loc, distance = doublebatch_projection_placement_distance_torch(xy_grid, start_point, end_point)

    # Determine valid locations and calculate weights
    valid_mask = (loc >= 0) & (loc <= 1)
    weight = gaussian_weight_torch(distance, variance)
    weight *= valid_mask

    # Apply occlusion
    occlusion += magnitude * weight

    return occlusion

def all_occlusions(X, dims, variance=1.):
    """X is the symmetric matrix of chord weights.
    
    For the sake of being image-quality agnostic, we scale all considerations of distance by the width of the image, so x and y are always in the range [0, 1]."""
    assert X.shape[0] == X.shape[1], "X must be a square matrix."

    occlusion = torch.zeros(dims)

    points = points_on_inscribed_circle(X.shape[0], 1.)

    for i in tqdm(range(X.shape[1])):
        for j in range(i+1, X.shape[1]):
            if X[i, j] > 0:
                occlusion = add_line_occlusion(occlusion, X[i, j], points[i], points[j], variance=variance)

    return occlusion

def vectorized_all_occlusions(X, dims, variance=1.):
    assert X.shape[0] == X.shape[1], "X must be a square matrix."

    occlusion = torch.zeros(dims)
    points = points_on_inscribed_circle(X.shape[0], dims[0])

    # Generate all combinations of points
    combinations = torch.combinations(torch.arange(X.shape[0]), r=2)
    start_indices = combinations[:, 0]
    end_indices = combinations[:, 1]

    # Use index_select to gather start and end points
    start_points = torch.index_select(points, 0, start_indices)
    end_points = torch.index_select(points, 0, end_indices)
    magnitudes = X[start_indices, end_indices]

    # Filter out combinations with zero magnitude
    valid = magnitudes > 0 
    start_points = start_points[valid]
    end_points = end_points[valid]
    magnitudes = magnitudes[valid]

    # Compute occlusions for all line segments in a batched manner
    occlusion = batched_line_occlusions(occlusion, magnitudes, start_points, end_points, variance)

    return occlusion


def batched_line_occlusions(occlusion, magnitudes, start_points, end_points, variance=torch.tensor(0.005)):
    height, width = occlusion.shape
    y_grid, x_grid = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing='ij')
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)  # Shape: [height, width, 2]

    AB = end_points - start_points  # Shape: [num_lines, 2]
    AP = xy_grid.unsqueeze(2) - start_points  # Shape: [height, width, num_lines, 2]
    
    # Dot products and norms
    AB_dot_AB = torch.sum(AB**2, dim=1)  # Shape: [num_lines]
    AP_dot_AB = torch.sum(AP * AB, dim=-1)  # Shape: [height, width, num_lines]
    
    t = AP_dot_AB / AB_dot_AB  # Shape: [height, width, num_lines]
    projection = start_points + t.unsqueeze(-1) * AB  # Shape: [height, width, num_lines, 2]
    
    distance = torch.norm(projection - xy_grid.unsqueeze(2), dim=-1)  # Shape: [height, width, num_lines]

    # Calculate Gaussian weights
    weights = gaussian_weight_torch(distance, variance)  # Shape: [height, width, num_lines]

    # Apply weights to each line's occlusion magnitude
    weighted_magnitudes = weights * magnitudes  # Shape: [height, width, num_lines]

    # Sum the contributions from all lines
    occlusion += torch.sum(weighted_magnitudes, dim=-1)

    return occlusion


def loss(X, img, variance=torch.tensor(0.005)):
    """X is the symmetric matrix of chord weights, img is the target image.
    
    For the sake of being image-quality agnostic, we scale all considerations of distance by the width of the image, so x and y are always in the range [0, 1]."""
    assert X.shape[0] == X.shape[1], "X must be a square matrix."
    assert img.shape[0] == img.shape[1], "Image must be square."
    
    # the occlusion matrix represents what the current set of chords would look like if they were drawn on a blank canvas
    occlusion = torch.zeros_like(img)
    
    points = points_on_inscribed_circle(X.shape[0], 1.)

    # for i in range(X.shape[1]):
    #     for j in range(i, X.shape[1]):
    #         if X[i, j] > 0:
    #             occlusion = add_line_occlusion(occlusion, X[i, j], points[i], points[j], variance)
    occlusion = all_occlusions(X, img.shape, variance=variance)

    torch.sigmoid_(occlusion)
    # print(occlusion)

    return torch.sum((img - occlusion)**2), occlusion

def vectorized_loss(X, img, variance=torch.tensor(0.005)):
    n = X.shape[0]
    assert X.shape == (n, n), "X must be a square matrix."
    assert img.shape[0] == img.shape[1], "Image must be square."

    occlusion = torch.zeros_like(img)
    points = points_on_inscribed_circle(n, img.shape[0])

    # Prepare tensors for all start and end points based on X
    combinations = torch.combinations(torch.arange(n), r=2)
    start_points = points[combinations[:, 0]]
    end_points = points[combinations[:, 1]]
    magnitudes = X[combinations[:, 0], combinations[:, 1]]

    # Filter out combinations with zero magnitude
    valid = magnitudes > 0 
    start_points = start_points[valid]
    end_points = end_points[valid]
    magnitudes = magnitudes[valid]

    # Batch compute occlusions
    occlusion = batched_line_occlusions(occlusion, magnitudes, start_points, end_points, variance)

    # Compute the loss
    return torch.sum((img - occlusion)**2), occlusion

def plot_diff(img, occlusions):
    fig, ax = plt.subplots(1, 3)

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("True Image")
    ax[0].axis('off')

    ax[1].imshow(occlusions.detach(), cmap='gray')
    ax[1].set_title("Occlusion")
    ax[1].axis('off')

    ax[2].imshow(img - occlusions.detach(), cmap='gray')
    ax[2].set_title("Difference")
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()

# %%
if __name__ == "__main__":
    # Test loss function
    import matplotlib.pyplot as plt
    from representation import DiscreteChordRepresentation
    from read_scale import load_image
    import numpy as np

    img = torch.tensor(load_image("et.png", square=True), dtype=torch.float32)
    X = torch.rand((32, 32), dtype=torch.float32)
    X = (X + X.T) / 2

    print(X)
    # occlusions = all_occlusions(X, img.shape, variance=torch.tensor(0.005))
    # occlusions = torch.zeros_like(img)
    # occlusions = batched_line_occlusions(occlusions, torch.tensor(1., dtype=torch.float32), torch.tensor([.25, 0.25], dtype=torch.float32), torch.tensor([0.75, 0.75], dtype=torch.float32), variance=torch.tensor(0.005))
    loss, occlusions = loss(X, img)
    print(loss)

    # torch.sigmoid_(occlusions)
    print(torch.max(occlusions))

    print(vectorized_loss(X, img, variance=torch.tensor(0.005)))

    plot_diff(img, occlusions)
# %%
