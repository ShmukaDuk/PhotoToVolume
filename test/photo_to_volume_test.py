import unittest
import cv2

import sys
import os

# Get the parent directory of the current file (test.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

# Now you can import modules from the src directory
from src.photo_to_volume import CalculateVolume

# Rest of your unit test code here

class TestCalculateVolume(unittest.TestCase):
    def test_resize_image(self):
        # Create an instance of your CalculateVolume class
        calculator = CalculateVolume(image_path='test/data/kfc_pot.jpg')

        # Load a test image
        test_image = cv2.imread('test/data/kfc_pot.jpg')

        # Define a target size
        target_size = (500, 500)

        # Resize the image using the function you want to test
        resized_image = calculator.resize_image(test_image, target_size)

        # Get the dimensions of the resized image
        resized_height, resized_width, _ = resized_image.shape

        # Calculate the expected dimensions while maintaining the aspect ratio
        expected_width = min(target_size[0], int(target_size[1] * (test_image.shape[1] / test_image.shape[0])))
        expected_height = min(target_size[1], int(target_size[0] * (test_image.shape[0] / test_image.shape[1])))

        # Assert that the resized image dimensions match the expected dimensions
        self.assertEqual(resized_width, expected_width)
        self.assertEqual(resized_height, expected_height)

if __name__ == '__main__':
    unittest.main()
