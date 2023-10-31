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
    def setUp(self):
        # Create an instance of your CalculateVolume class
        self.calculator = CalculateVolume(image_path=None, pot_width=None, pot_height=None, poly_accuracy='10')  # Initialize with a placeholder image_path

    def test_resize_image(self):
        # List all files ending with .jpg in the data/ directory
        data_dir = 'test/data'
        jpg_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]

        for jpg_file in jpg_files:
            print("Testing: ",  jpg_file)
            # Load a test image
            test_image = cv2.imread(os.path.join(data_dir, jpg_file))

            # Define a target size
            target_size = (500, 500)

            # Resize the image using the function you want to test
            resized_image = self.calculator.resize_image(test_image, target_size)

            # Get the dimensions of the resized image
            resized_height, resized_width, _ = resized_image.shape

            # Calculate the expected dimensions while maintaining the aspect ratio
            expected_width = min(target_size[0], int(target_size[1] * (test_image.shape[1] / test_image.shape[0])))
            expected_height = min(target_size[1], int(target_size[0] * (test_image.shape[0] / test_image.shape[1])))

            # Assert that the resized image dimensions match the expected dimensions
            self.assertEqual(resized_width, expected_width)
            self.assertEqual(resized_height, expected_height)
            out_dir = 'test/out'
            resized_filename = os.path.splitext(jpg_file)[0] + '_resized.jpg'
            output_path = os.path.join(out_dir, resized_filename)
            cv2.imwrite(output_path, resized_image)


    def test_calc_volume(self):
        data_dir = 'test/data'
        jpg_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
        for jpg_file in jpg_files:
            print('file name: ')
            print(jpg_file)
            pot_height_mm = int(jpg_file.split('_')[-1].split("x")[1])
            pot_width_mm = int(jpg_file.split('_')[-1].split("x")[0])
            pot_volume_l = int(jpg_file.split('_')[-1].split("x")[2].replace('l.jpg',''))
            
            print( pot_height_mm, pot_width_mm, pot_volume_l)

            
            calculator = CalculateVolume(os.path.join(data_dir, jpg_file), pot_width_mm/10, pot_height_mm/10, '10')
            calculator.prep_image()
            volume_litres = calculator.calculate_volume()
            print('POT: ',jpg_file, "Volume in Liters:",  volume_litres)
            calculator.create_plots()

if __name__ == '__main__':
    unittest.main()
