import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import interp1d
from scipy.integrate import trapz, quad
from rembg import remove


POT_HEIGHT_CM = 33
POT_WIDTH_CM = 33

CM_TO_MM = 10
MM3_TO_CM3 = 0.001
CM3_TO_L = 0.001

def prep_image():
    # Store path of the image in the variable input_path
    input_path = 'images/pot.jpg'

    # Store path of the output image in the variable output_path
    output_path = 'images/cropped_pot.jpg'

    # Processing the image
    input = Image.open(input_path)

    # Removing the background from the given Image
    output = remove(input)

    # Convert the RGBA image to RGB mode
    output_rgb = output.convert("RGB")

    # Save the converted image as JPEG
    output_rgb.save(output_path)
    #Saving the image in the given path



def resize_image(image, target_size):
    aspect_ratio = image.shape[1] / image.shape[0]
    target_width = int(target_size[0])
    target_height = int(target_width / aspect_ratio)
    if target_height > target_size[1]:
        target_height = target_size[1]
        target_width = int(target_height * aspect_ratio)
    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    return resized_image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

def filter_sort_data_points(data_points):
    leftmost_idx = np.argmin(data_points[:, 0])
    leftmost_point = data_points[leftmost_idx]
    rightmost_idx = np.argmax(data_points[:, 0])
    rightmost_point = data_points[rightmost_idx]
    lowest_x = np.min(data_points[:, 0])
    lowest_y = np.min(data_points[:, 1])
    data_points[:, 0] -= lowest_x
    data_points[:, 1] -= lowest_y
    mid_point = ((rightmost_point[0] - leftmost_point[0]) / 2) + leftmost_point[0]
    mask = data_points[:, 0] <= mid_point
    filtered_data_points = data_points[mask]
    highest_y = np.max(filtered_data_points[:, 1])
    threshold_low = highest_y / 20
    lpf_mask = filtered_data_points[:, 1] >= threshold_low
    filtered_data_points = filtered_data_points[lpf_mask]
    threshold_high = highest_y - (highest_y / 20)
    mask = filtered_data_points[:, 1] < threshold_high
    filtered_data_points = filtered_data_points[mask]
    sorted_data_points = filtered_data_points[np.argsort(filtered_data_points[:, 0])]
    return sorted_data_points

def interpolate_flip_data_points(sorted_data_points):
    x_coords = sorted_data_points[:, 0]
    y_coords = sorted_data_points[:, 1]
    interp_func = interp1d(y_coords, x_coords)
    desired_y_values = np.arange(min(y_coords), max(y_coords), 5)
    interpolated_x_values = interp_func(desired_y_values)
    interpolated_points = np.column_stack((interpolated_x_values, desired_y_values))
    highest_y = np.max(sorted_data_points[:, 1])
    interpolated_points_flipped = highest_y - interpolated_points
    return interpolated_points_flipped

def scale_data_points(interpolated_points_flipped, pot_height_cm, pot_width_cm, cm_to_mm):
    highest_y_mm = np.max(interpolated_points_flipped[:, 0])
    highest_x_mm = np.max(interpolated_points_flipped[:, 1])
    y_ratio = ((pot_width_cm / 2) / highest_y_mm) * cm_to_mm
    x_ratio = (pot_height_cm / highest_x_mm) * cm_to_mm
    scaled_data_points_mm = []
    for point in interpolated_points_flipped:
        scaled_point = []
        if point[0] != 0:
            scaled_point.append(int(point[0] * y_ratio))
        if point[1] != 0:
            scaled_point.append(int(point[1] * x_ratio))
        scaled_data_points_mm.append(scaled_point)
    data_points_mm = np.array(scaled_data_points_mm)
    return data_points_mm

def calculate_volume(data_points_mm, degree, pot_height_cm, pot_width_cm):
    x_coords_mm = data_points_mm[:, 1]
    y_coords_mm = data_points_mm[:, 0]
    coefficients = np.polyfit(x_coords_mm, y_coords_mm, degree)
    polynomial = np.poly1d(coefficients)
    x_values = np.linspace(min(x_coords_mm), max(x_coords_mm), 10)
    y_values = polynomial(x_values)
    area = trapz(y_values, x_values)
    volume, error = quad(lambda x: np.pi * polynomial(x)**2, min(x_coords_mm), max(x_coords_mm))
    volume_cm3 = volume * MM3_TO_CM3
    litres = volume_cm3 * CM3_TO_L
    return litres

def calculate_volume_from_image():
    # Set constants
    isolated_image_path = 'images/cropped_pot.jpg'

    # Read the image
    image = cv2.imread(isolated_image_path)

    # Resize the image
    target_size = (1000, 1000)
    resized_image = resize_image(image, target_size)

    # Preprocess image and extract data points
    data_points = preprocess_image(resized_image)

    # Filter and sort data points
    sorted_data_points = filter_sort_data_points(data_points)

    # Interpolate and flip data points
    interpolated_points_flipped = interpolate_flip_data_points(sorted_data_points)

    # Scale data points
    data_points_mm = scale_data_points(interpolated_points_flipped, POT_HEIGHT_CM, POT_WIDTH_CM, CM_TO_MM)

    # Calculate volume
    degree = 5
    volume_litres = calculate_volume(data_points_mm, degree, POT_HEIGHT_CM, POT_WIDTH_CM)

    return volume_litres

# Example usage


prep_image()
volume_litres = calculate_volume_from_image()
print("Volume in Liters:", volume_litres)
