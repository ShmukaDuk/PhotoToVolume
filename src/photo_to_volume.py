import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import interp1d
from scipy.integrate import trapz, quad
from rembg import remove
import matplotlib.pyplot as plt

class CalculateVolume:
    def __init__(self, image_path, pot_width, pot_height):
        self.POT_HEIGHT_CM = pot_height
        self.POT_WIDTH_CM = pot_width
        self.CM_TO_MM = 10
        self.MM3_TO_CM3 = 0.001
        self.CM3_TO_L = 0.001
        self.OG_PHOTO = image_path  # Use the provided image_path
        # self.image_data = NULL
        

    def prep_image(self):
        # Processing the image
        input = Image.open(self.OG_PHOTO)

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

        data_points = []

        for contour in contours:
            arc_length = cv2.arcLength(contour, True)
            min_arc_length_threshold = 100
            if arc_length > min_arc_length_threshold:
                for point in contour:
                    x, y = point[0]
                    data_points.append((x, y))

        data_points = np.array(data_points)

        return data_points

    def filter_sort_data_points(self, data_points):
        # Find the leftmost and rightmost points
        leftmost_point = data_points[data_points[:, 0].argmin()]
        rightmost_point = data_points[data_points[:, 0].argmax()]

        # Find the lowest x and y values
        lowest_x = data_points[:, 0].min()
        lowest_y = data_points[:, 1].min()

        # Translate data points to the origin
        data_points[:, 0] -= lowest_x
        data_points[:, 1] -= lowest_y

        # Calculate the midpoint between leftmost and rightmost points
        mid_point = (rightmost_point[0] - leftmost_point[0]) / 2 + leftmost_point[0]

        # Create a mask to filter data points based on x-coordinate
        mask = data_points[:, 0] <= mid_point
        filtered_data_points = data_points[mask]

        # Calculate the highest y value in filtered data points
        highest_y = filtered_data_points[:, 1].max()

        # Define thresholds for filtering
        threshold_low = highest_y / 20
        threshold_high = highest_y - (highest_y / 20)

        # Apply additional filtering based on y-coordinate
        lpf_mask = (filtered_data_points[:, 1] >= threshold_low) & (filtered_data_points[:, 1] < threshold_high)
        filtered_data_points = filtered_data_points[lpf_mask]

        # Sort and remove duplicates
        sorted_data_points = np.unique(
            filtered_data_points[~np.isnan(filtered_data_points[:, 0])][np.argsort(filtered_data_points[:, 0])],
            axis=0)

        return sorted_data_points


    def interpolate_flip_data_points(self, sorted_data_points):
        _, unique_indices = np.unique(sorted_data_points[:, 1], return_index=True)
        sorted_data_points = sorted_data_points[unique_indices]
        x_coords = sorted_data_points[:, 0]
        y_coords = sorted_data_points[:, 1]

        interp_func = interp1d(y_coords, x_coords)

        desired_y_values = np.arange(min(y_coords), max(y_coords), 5)
        interpolated_x_values = interp_func(desired_y_values)
        interpolated_points = np.column_stack((interpolated_x_values, desired_y_values))
        highest_y = np.max(sorted_data_points[:, 1])
        interpolated_points_flipped = highest_y - interpolated_points
        interpolated_points_flipped = interpolated_points_flipped[~np.isnan(interpolated_points_flipped).any(axis=1)]

        return interpolated_points_flipped

    def scale_data_points(self, interpolated_points_flipped):
        filtered_points = []

        for point in interpolated_points_flipped:
            # Check if any value in the point is NaN using numpy's isnan function
            if np.isnan(point).any():
                continue  # Skip this point if it contains NaN values
            # Add the point to the filtered points list if it doesn't contain NaN values
            filtered_points.append(point)
        filtered_points = np.array(filtered_points)

        highest_y_mm = np.max(filtered_points[:, 0])
        highest_x_mm = np.max(filtered_points[:, 1])
        y_ratio = ((self.POT_WIDTH_CM / 2) / highest_y_mm) * self.CM_TO_MM
        x_ratio = (self.POT_HEIGHT_CM / highest_x_mm) * self.CM_TO_MM
        scaled_data_points_mm = []
        for point in filtered_points:
            scaled_point = []
            if point[0] != 0:
                scaled_point.append(int(point[0] * y_ratio))
            if point[1] != 0:
                scaled_point.append(int(point[1] * x_ratio))
            scaled_data_points_mm.append(scaled_point)
        data_points_mm = np.array(scaled_data_points_mm)
        return data_points_mm

    def generate_polynomial(self, data_points_mm):
        degree = 10
        x_coords_mm = data_points_mm[:, 1]
        y_coords_mm = data_points_mm[:, 0]
        coefficients = np.polyfit(x_coords_mm, y_coords_mm, degree)
        polynomial = np.poly1d(coefficients)
        return polynomial
    def calculate_volume(self):
       
        # Resize the image
        target_size = (1000, 1000)
        resized_image = self.resize_image(self.no_bg_image, target_size)

        # Preprocess image and extract data points
        data_points = self.preprocess_image(resized_image)

        # Filter and sort data points
        sorted_data_points = self.filter_sort_data_points(data_points)

        # Interpolate and flip data points
        interpolated_points_flipped = self.interpolate_flip_data_points(sorted_data_points)

        # Scale data points
        data_points_mm = self.scale_data_points(interpolated_points_flipped)

        poly = self.generate_polynomial(data_points_mm)

        volume_litres = self.calculate_volume_from_poly(poly, data_points_mm)

        return volume_litres

    def calculate_volume_from_poly(self, poly, data_points_mm):
        x_coords_mm = data_points_mm[:, 1]
        x_values = np.linspace(min(x_coords_mm), max(x_coords_mm), 10)
        y_values = poly(x_values)
        area = trapz(y_values, x_values)
        volume, error = quad(lambda x: np.pi * poly(x) ** 2, min(x_coords_mm), max(x_coords_mm))
        volume_cm3 = volume * self.MM3_TO_CM3
        litres = volume_cm3 * self.CM3_TO_L
        return litres    
    


# Example usage
if __name__ == "__main__":
    calculator = CalculateVolume()
    calculator.prep_image()
    volume_litres = calculator.calculate_volume()
    print("Volume in Liters:", volume_litres)
  