from PIL import Image
import numpy as np

# Load the image
image_path = 'et.png'
image = Image.open(image_path)

# Convert the image to grayscale
gray_image = image.convert('L')

# Convert the grayscale image to a numpy array
pixel_array = np.array(gray_image)

# Print the pixel array
print(pixel_array)
