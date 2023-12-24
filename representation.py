# %%
import math
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

from read_scale import load_image

# %%
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

def path_integral(matrix, start_point, end_point, variance):
    num_points = int(euclidean(start_point, end_point)) * 2
    line_points = interpolate_line_segment(np.array(start_point), np.array(end_point), num_points)

    integral = 0
    for point in line_points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < matrix.shape[1] and 0 <= y < matrix.shape[0]:
            distance_to_line = perpendicular_distance_and_projection(point, start_point, end_point)
            if distance_to_line is not None:
                darkness = matrix[y, x]
                weight = gaussian_weight(distance_to_line, variance)
                integral += darkness * weight

    return integral

def contribution(matrix_shape, start_point, end_point, variance):
    num_points = int(euclidean(start_point, end_point)) * 2
    line_points = interpolate_line_segment(np.array(start_point), np.array(end_point), num_points)

    contribution_matrix = np.zeros(matrix_shape)

    for point in line_points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < matrix_shape[1] and 0 <= y < matrix_shape[0]:
            distance_to_line = perpendicular_distance_and_projection(point, start_point, end_point)
            if distance_to_line is not None:
                weight = gaussian_weight(distance_to_line, variance)
                contribution_matrix[y, x] += weight

    return contribution_matrix


def darken_by_contribution(matrix, start_point, end_point, variance):
    num_points = int(euclidean(start_point, end_point)) * 2
    line_points = interpolate_line_segment(np.array(start_point), np.array(end_point), num_points)

    modified_matrix = np.copy(matrix)

    for point in line_points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < modified_matrix.shape[1] and 0 <= y < modified_matrix.shape[0]:
            distance_to_line = perpendicular_distance_and_projection(point, start_point, end_point)
            if distance_to_line is not None:
                weight = gaussian_weight(distance_to_line, variance)
                # Darken the pixel by reducing its value
                modified_matrix[y, x] = max(modified_matrix[y, x] - weight, 0)

    return modified_matrix

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

# %%
class DiscreteChordRepresentation:
    def __init__(self, img_path="et.png", num_pegs=64, thread_darkness = 1.0, variance=0.05):
        self.img = 1 - load_image(img_path)
        self.num_pegs = num_pegs
        self.thread_darkness = thread_darkness
        self.variance = variance
        self.dims = self.img.shape

        self.matrix = np.zeros(self.dims) # for tracking darkness updates, subtracted from image for updated image

        self.pegs = [point_on_inscribed_circle(self.dims[0], self.dims[1], theta) for theta in np.linspace(0, 2*math.pi, num_pegs, endpoint=False)]
        self.chords = np.zeros((num_pegs, num_pegs)) # if a chord from a to b is included, chords[a, b] = 1

    def add_chord(self, start_peg, end_peg):
        if self.chords[start_peg, end_peg] == 0:  # Check if chord has not been added yet
            start_point = self.pegs[start_peg]
            end_point = self.pegs[end_peg]

            self.matrix += contribution(self.matrix.shape, start_point, end_point, self.variance)
            self.chords[start_peg, end_peg] = 1  # Mark the chord as added
            self.chords[end_peg, start_peg] = 1  # Since chord is bi-directional

    def set_chords(self, chords):
        self.chords = chords

    def get_updated_image(self):
        # Ensure the darkness doesn't cause values to go negative
        updated_img = np.clip(self.img + self.matrix, 0, 1)
        return updated_img

    def visualize_current_state(self):
        updated_img = self.get_updated_image()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ax = axes.ravel()

        ax[0].imshow(self.img, cmap='gray')
        ax[0].set_title("Original Image")
        ax[0].axis('off')

        ax[1].imshow(1 - self.matrix, cmap='gray')
        ax[1].set_title("Darkness Contribution")
        ax[1].axis('off')

        ax[2].imshow(updated_img, cmap='gray')
        ax[2].set_title("Updated Image")
        ax[2].axis('off')

        plt.tight_layout()
        plt.show()

    def visualize_chords(self):
        # Plot the circle
        pegs = np.array(self.pegs)

        fig, ax = plt.subplots(1, 2)

        ax[0].plot(pegs[:,0], pegs[:,1], 'o')
        ax[0].set_aspect('equal')

        # Plot the chords
        for i in range(len(self.pegs)):
            for j in range(i+1, len(self.pegs)):
                ax[0].plot([pegs[i][0], pegs[j][0]], [pegs[i][1], pegs[j][1]], 'k-', alpha=self.chords[i, j])

        processed_img = 255 * self.img - 255

        ax[1].imshow(processed_img, cmap='gray')
        ax[1].set_title("True Image")
        ax[1].axis('off')

        plt.tight_layout()
        plt.show()


# %%
if __name__ == "__main__":
    dcr = DiscreteChordRepresentation()

    dcr.add_chord(0, 32)
    dcr.visualize_current_state()
# %%
