import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import interp1d
from scipy.integrate import trapz, quad
from rembg import remove
import matplotlib.pyplot as plt
from scipy.integrate import trapz

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # Add this import
from mpl_toolkits.mplot3d import Axes3D  # Importing the necessary module

class CalculateVolume:
    def __init__(self, image_path, pot_width, pot_height, poly_accuracy):
        self.POT_HEIGHT_CM = pot_height
        self.POT_WIDTH_CM = pot_width
        self.CM_TO_MM = 10
        self.MM3_TO_CM3 = 0.001
        self.CM3_TO_L = 0.001
        self.OG_PHOTO = image_path  # Use the provided image_path
        self.poly_accuracy = int(poly_accuracy)
        self.pot_name = image_path
        # self.image_data = NULL
        
                
    def create_plots(self):
            # Create some sample data
            x = np.linspace(0, 2 * np.pi, 100)
            y = np.sin(x)

            # Create a figure with multiple subplots
            self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))           

            # Display self.no_bg_image in a subplot
            ax_image = self.axes[0, 0]  # Adjust the row and column as needed
            ax_image.imshow(self.origional_image)
            ax_image.set_title('Image Plot')
            ax_image = self.axes[0, 1]  # Adjust the row and column as needed
            ax_image.imshow(self.no_bg_image)
            ax_image.set_title('Image Plot')
            ax_image = self.axes[0, 2]  # Adjust the row and column as needed
            ax_image.imshow(self.resized_image)
            ax_image.set_title('Resized image')
            ax_image = self.axes[1, 0]  # Adjust the row and column as needed
            x_coords_mm = [point[0] for point in self.contour_points]
            y_coords_mm = [point[1] for point in self.contour_points]
            # Create a scatter plot
            ax_image.scatter(x_coords_mm, y_coords_mm, c='b', marker='o', label='Scaled Data Points')
            
            
            #plot poly
            ax_image = self.axes[1, 1]  # Adjust the row and column as needed
            
            x_coords = self.sorted_poly_points[:, 0]
            y_coords = self.sorted_poly_points[:, 1]
            x_fit = np.linspace(min(x_coords), max(x_coords), 100)
            y_fit = self.polynomial(x_fit)
            ax_image.scatter(x_coords, y_coords, label='Data Points', color='blue')
            ax_image.plot(x_fit, y_fit, label=f'Fitted Polynomial (Degree {self.poly_accuracy})', color='red')
            
            # 3D plot



            plt.tight_layout()
            plt.show()
            
              
                
  


    def prep_image(self):
        # Processing the image
        input = Image.open(self.OG_PHOTO)
        if input.width > input.height:
            # Rotate the image by 90 degrees (clockwise)
            input = input.transpose(Image.ROTATE_270)
        self.origional_image = input
        # Removing the background from the given image
        output = remove(input)

        # Save the image with transparency as PNG
        self.no_bg_image = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR) 
        
        
        

    def resize_image(self, image, target_size):
        aspect_ratio = image.shape[1] / image.shape[0]
        target_width = int(target_size[0])
        target_height = int(target_width / aspect_ratio)
        if target_height > target_size[1]:
            target_height = target_size[1]
            target_width = int(target_height * aspect_ratio)
        resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        self.resized_image = resized_image
        return resized_image

    def preprocess_image(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the grayscale image to create a binary mask
        _, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the original image with 3 color channels (BGR)
        outlined_image = image.copy()

        # Draw the contours on the outlined image
        cv2.drawContours(outlined_image, contours, -1, (0, 255, 0), 2)
        # Convert BGR image to RGB format (Matplotlib uses RGB)
        outlined_image_rgb = cv2.cvtColor(outlined_image, cv2.COLOR_BGR2RGB)

        data_points = []

        for contour in contours:
                arc_length = cv2.arcLength(contour, True)
                min_arc_length_threshold = 100
                if arc_length > min_arc_length_threshold:
                    for point in contour:
                        x, y = point[0]
                        # Flip the Y-coordinate here
                        y = image.shape[0] - y
                        data_points.append((x, y))

        self.contour_points = data_points
        return data_points
    
    def process_data(self, data_from_image):
            data_points = np.array(data_from_image)
            split_points = self.split_points(data_points)
            rotated_points =  self.rotate_points(split_points)
            zerod_points =  self.zero_rotated_points(rotated_points)
            mm_points = self.scale_to_mm(zerod_points)
            zerod_points_clean = self.remove_duplicate_y(mm_points)
            filled_x_points = self.interpolate_points(zerod_points_clean)
            polynomial = self.fit_polynomial(filled_x_points)
            volume_litres = self.calculate_volume_from_polygon(polynomial)
           
            return volume_litres
       
       
    def split_points(self, raw_points):

        # Find the furthest left and right points
        leftmost_x = raw_points[:, 0].min()
        rightmost_x = raw_points[:, 0].max()

        # Calculate the halfway point
        halfway_x = (leftmost_x + rightmost_x) / 2

        # Filter the data points to keep only those on the left side (X <= halfway)
        right_points = raw_points[raw_points[:, 0] > halfway_x]
        return right_points

    def rotate_points(self, split_points):
         # Convert the angle from degrees to radians
        swapped_points = np.column_stack((split_points[:, 1], split_points[:, 0]))
        self.bot_left_point = swapped_points[swapped_points[:, 1].argmin()]
        print(self.bot_left_point)
        return swapped_points


    def zero_rotated_points(self, rotated_points):
        # Find the minimum X and Y values
        min_x = rotated_points[:, 0].min()
        min_y = rotated_points[:, 1].min()
        

        # Subtract the minimum X and Y values to shift all points
        zeroed_points = rotated_points - np.array([min_x, min_y])
        return zeroed_points
        
    def scale_to_mm(self, zeroed_points):
        # Calculate the scaling ratios for X and Y
        y_ratio = 5*  self.POT_WIDTH_CM / zeroed_points[:, 1].max()
        x_ratio = 10* self.POT_HEIGHT_CM / zeroed_points[:, 0].max()

        # Scale the points
        scaled_points = zeroed_points * np.array([x_ratio, y_ratio])

        return scaled_points

    def remove_duplicate_y(self, scaled_data_points):
        # Find unique Y values and their corresponding indices
        unique_y, unique_indices = np.unique(scaled_data_points[:, 1], return_index=True)

        # Select the points with unique Y values
        unique_points = scaled_data_points[unique_indices]

        return unique_points
        
        
        
    def interpolate_points(self, unique_points):
        # Sort the unique points based on the X-coordinate
        unique_points = unique_points[unique_points[:, 0].argsort()]

        # Extract the X and Y coordinates
        x_coords = unique_points[:, 0]
        y_coords = unique_points[:, 1]

        # Create an interpolation function with clipping
        interpolator = interp1d(x_coords, y_coords, kind='linear')

        # Define the range of x-values from 0 to the largest X-coordinate
        largest_x = int(np.max(x_coords))
        new_x_values = np.arange(0, largest_x + 1)

        # Interpolate the corresponding y-values for the new x-values with clipping
        new_y_values = np.clip(interpolator(new_x_values), y_coords.min(), y_coords.max())

        # Create a new array with the clipped interpolated points
        new_array = np.column_stack((new_x_values, new_y_values))

        return new_array
        
    
    def fit_polynomial(self, unique_points):
        # Sort the unique points based on the X-coordinate
        sorted_points = unique_points[unique_points[:, 0].argsort()]
        self.sorted_poly_points = sorted_points
        # Extract X and Y coordinates
        x_coords = sorted_points[:, 0]
        self.largest_point = (max(x_coords))
        y_coords = sorted_points[:, 1]

        # Define the degree of the polynomial (adjust as needed)
        degree = self.poly_accuracy  # You can change this degree as per your data

        # Fit a polynomial to the data
        coefficients = np.polyfit(x_coords, y_coords, degree)

        # Create a polynomial function from the coefficients
        polynomial = np.poly1d(coefficients)
        self.polynomial = polynomial
        # Create a range of X values for the fitted curve
        x_fit = np.linspace(min(x_coords), max(x_coords), 100)
        # Calculate the corresponding Y values using the polynomial function
        y_fit = polynomial(x_fit)
        
        self.polynomail = polynomial
        self.x_fit = np.linspace(min(x_coords), max(x_coords), 100)
        self.y_fit = polynomial(x_fit)

        return polynomial
    
        
    def calculate_volume_from_polygon(self, polygon):
       

        H = self.largest_point - 1   # Replace with your desired height

        # Number of intervals for integration
        num_intervals = 100

        # Initialize the volume
        volume = 0.0

        # Create a figure and a 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(num_intervals):
            h0 = i * H / num_intervals
            h1 = (i + 1) * H / num_intervals
            r0 = polygon(h0)
            r1 = polygon(h1)
            volume += (np.pi * r0**2 + np.pi * r1**2) / 2 * (h1 - h0)

            # Calculate the radius for this circle (use the larger of r0 and r1)
            radius = max(r0, r1)

            # Create vertices for the circle (spanning 360 degrees)
            theta = np.linspace(0, 2 * np.pi, 100)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = np.array([h0] * 100)

            # Plot the circle
            ax.plot(x, y, z, color='c')

        # Set axis labels
        ax.set_xlabel('Width (mm)')
        ax.set_ylabel('Width (mm)')
        ax.set_zlabel('Height (mm)')

        # Set aspect ratio to 'auto' for better visualization
        ax.set_box_aspect([1, 1, 1])

        # Set limits for the axes
        ax.set_xlim(-500, 500)
        ax.set_ylim(-500, 500)
        ax.set_zlim(-500, 500)
        plt.gca().set_aspect('equal', adjustable='box')

        
        print("Volume of the cylinders:", volume / 1000000)

        return volume / 1000000

    def calculate_volume(self):
       
        # Resize the image
        target_size = (1000, 1000)
        resized_image = self.resize_image(self.no_bg_image, target_size)

        # Preprocess image and extract data points
        data_points = self.preprocess_image(resized_image)


        volume_litres = self.process_data(data_points)

        return volume_litres

   
# Example usage
if __name__ == "__main__":
    calculator = CalculateVolume("C:/dev/git/photo-to-volume/test/data/black_pot_276x276x11l.jpg", 32, 37, 10)
    calculator.prep_image()
    volume_litres = calculator.calculate_volume()
    calculator.create_plots()
    print("Volume in Liters:", volume_litres)
  