
from PIL import Image

from rembg import remove


def prep_image():
    # Store path of the image in the variable input_path
    input_path = 'fridge.jpg'

    # Store path of the output image in the variable output_path
    output_path = 'fridge_cropped.png'  # Save as PNG for transparent background

    # Processing the image
    input = Image.open(input_path)

    # Removing the background from the given image
    output = remove(input)

    # Save the image with transparency as PNG
    output.save(output_path)
prep_image()