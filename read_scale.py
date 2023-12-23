from PIL import Image
import numpy as np

def load_image(image_path='et.png'):
    # Load the image
    image = Image.open(image_path)

    # Ensure image has an alpha (transparency) channel
    image = image.convert("RGBA")

    # Create an image object for the output image
    gray_image = Image.new("L", image.size)

    # Convert each pixel to grayscale
    for x in range(image.width):
        for y in range(image.height):
            r, g, b, a = image.getpixel((x, y))
            if a == 0:
                gray = 255  # Makes transparent parts white
            else:
                gray = int(0.299*r + 0.587*g + 0.114*b)
            gray_image.putpixel((x, y), gray)

    # Convert the grayscale image to a numpy array
    pixel_array = np.array(gray_image)

    # Calculate the dimensions of the original array
    height, width = pixel_array.shape

    # Determine the size of the square subset
    size = min(height, width)

    # Calculate the starting indices for the subset
    start_row = (height - size) // 2
    start_col = (width - size) // 2

    # Create the square subset array
    subset_array = pixel_array[start_row:start_row+size, start_col:start_col+size]

    # Invert the array (so black is white and vice versa)
    # and scale it to the range 0-1
    invert_scaled_array = (255-subset_array)/255

    return invert_scaled_array