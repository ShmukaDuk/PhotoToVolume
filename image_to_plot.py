import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import interp1d
from scipy.integrate import trapz, quad
from rembg import remove
import matplotlib.pyplot as plt



POT_HEIGHT_CM = 20
POT_WIDTH_CM = 20

CM_TO_MM = 10
MM3_TO_CM3 = 0.001
CM3_TO_L = 0.001

def prep_image():


    # Processing the image
    input = Image.open(OG_PHOTO)

    # Removing the background from the given image
    output = remove(input)

    # Save the image with transparency as PNG
    output.save(POT_NO_BG)


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

    # Save the outlined image as PNG with transparency
    cv2.imwrite(POT_WITH_CONTOURS, outlined_image)

    data_points = []

    for contour in contours:
        arc_length = cv2.arcLength(contour, True)
        min_arc_length_threshold = 100
        if arc_length > min_arc_length_threshold:
            for point in contour:
                x, y = point[0]
                data_points.append((x, y))

    data_points = np.array(data_points)

    lowest_x = np.min(data_points[:, 0])
    lowest_y = np.min(data_points[:, 1])
    highest_x = np.max(data_points[:, 0])
    highest_y = np.max(data_points[:, 1])
    print("Lowest x:", lowest_x)
    print("Lowest y:", lowest_y)
    print("Highest x:", highest_x)
    print("Highest y:", highest_y)

    image_pot_width = highest_x - lowest_x
    image_pot_height = highest_y - lowest_y
    scale_x = (POT_WIDTH_CM / image_pot_width) * 10
    scale_y = (POT_HEIGHT_CM / image_pot_height) * 10
    scaled_image = cv2.resize(image, (int(image.shape[1] * scale_x), int(image.shape[0] * scale_y)), interpolation=cv2.INTER_LINEAR)

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
    print(highest_x_mm, highest_y_mm)
    y_ratio = ((pot_width_cm / 2) / highest_y_mm) * cm_to_mm
    x_ratio = (pot_height_cm / highest_x_mm) * cm_to_mm
    print("getratiod: ", x_ratio, y_ratio)
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

def generate_polynomial(data_points_mm):
    degree = 100
    x_coords_mm = data_points_mm[:, 1]
    y_coords_mm = data_points_mm[:, 0]
    coefficients = np.polyfit(x_coords_mm, y_coords_mm, degree)
    polynomial = np.poly1d(coefficients)
    # Draw circles on the image
        
    # todo plot poly over the scaled image


    return polynomial

def calculate_volume(polynomial, data_points_mm):
    x_coords_mm = data_points_mm[:, 1]
    y_coords_mm = data_points_mm[:, 0]
    for point in data_points_mm:
        print(point)
    x_values = np.linspace(min(x_coords_mm), max(x_coords_mm), 10)
    y_values = polynomial(x_values)
    area = trapz(y_values, x_values)
    volume, error = quad(lambda x: np.pi * polynomial(x)**2, min(x_coords_mm), max(x_coords_mm))
    volume_cm3 = volume * MM3_TO_CM3
    litres = volume_cm3 * CM3_TO_L
    plot_polynomial(polynomial, min(x_coords_mm), max(x_coords_mm), data_points_mm)
    return litres


def plot_polynomial(polynomial, x_coords_min, x_coords_max, data_points_mm):
    x = np.linspace(x_coords_min, x_coords_max, 100)
    y = polynomial(x)

    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the polynomial
    ax.plot(x, y, color='blue', label='Polynomial')

    # Plot the data points
    x_data = data_points_mm[:, 1]
    y_data = data_points_mm[:, 0]
    ax.scatter(x_data, y_data, color='red', label='Data Points')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Polynomial Plot')

    # Set the aspect ratio to equal
    ax.set_aspect('equal')

    # Adjust the plot dimensions for equal height and y-axis starting at 0
    y_range = max(y_data) - min(y_data)
    x_range = x_coords_max - x_coords_min
    ratio = y_range / x_range
    ax.set_xlim([x_coords_min, x_coords_max])
    ax.set_ylim([0, max(y_data) + ratio * x_range / 2])

    # Show a legend
    ax.legend()

    # Show the plot
    plt.show()



def calculate_volume_from_image():

    # Read the image
    image = cv2.imread(POT_NO_BG)

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
    
    poly = generate_polynomial(data_points_mm)
    
    volume_litres = calculate_volume(poly, data_points_mm)
    
    return volume_litres

# Example usage


prep_image()
volume_litres = calculate_volume_from_image()
print("Volume in Liters:", volume_litres)
