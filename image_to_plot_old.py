import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw# Read the image
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from scipy.integrate import quad
# Read the image using cv2.imread
POT_HEIGHT_CM = 33
POT_WIDTH_CM = 33
CM_TO_MM = 10

image = cv2.imread('images/image_resized_no_bg.jpg')

# Set the target size
target_size = (1000, 1000)

# Calculate the aspect ratio of the original image
aspect_ratio = image.shape[1] / image.shape[0]

# Calculate the target size with the same aspect ratio
target_width = int(target_size[0])
target_height = int(target_width / aspect_ratio)
if target_height > target_size[1]:
    target_height = target_size[1]
    target_width = int(target_height * aspect_ratio)

# Resize the image using cv2.resize with INTER_LANCZOS4 interpolation
resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

# Create a black canvas of the target size
canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

# Calculate the position to paste the resized image
left = (target_size[0] - target_width) // 2
top = (target_size[1] - target_height) // 2

# Paste the resized image onto the canvas
canvas[top:top+target_height, left:left+target_width] = resized_image

# Save the resulting image with the correct file extension
cv2.imwrite('image_resized.jpg', canvas)

# Display the resulting image






image = cv2.imread('image_resized.jpg')


# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract data points from the contours that form a curve
data_points = []
for contour in contours:
    # Calculate the arc length of the contour
    arc_length = cv2.arcLength(contour, True)

    # Set a threshold for the minimum arc length to consider it as a curve
    min_arc_length_threshold = 100

    if arc_length > min_arc_length_threshold:
        for point in contour:
            x, y = point[0]
            data_points.append((x, y))
            


data_points = np.array(data_points)



# Find the index of the point on the far left (minimum x-coordinate)
leftmost_idx = np.argmin(data_points[:, 0])
leftmost_point = data_points[leftmost_idx]

# Find the index of the point on the far right (maximum x-coordinate)
rightmost_idx = np.argmax(data_points[:, 0])
rightmost_point = data_points[rightmost_idx]




# Find the x value of the point with the lowest x-coordinate
lowest_x = np.min(data_points[:, 0])

# Subtract the x value of the lowest point from all x-coordinates
data_points[:, 0] -= lowest_x



# Find the x value of the point with the lowest x-coordinate
lowest_y = np.min(data_points[:, 1])

# Subtract the x value of the lowest point from all x-coordinates
data_points[:, 1] -= lowest_y
# Draw the data points on the image
for point in data_points:
    cv2.circle(image, point, 5, (0, 0, 255), -1)  # Draw a red circle at each point



mid_point = ((rightmost_point[0] - leftmost_point[0]) / 2) + leftmost_point[0]
mask = data_points[:, 0] <= mid_point

# Apply the mask to filter the data_points array
filtered_data_points = data_points[mask]



#remove low 5%
filtered_data_points = np.array(filtered_data_points)
highest_y = np.max(filtered_data_points[:, 1])
threshold_low = highest_y / 20
lpf_mask = filtered_data_points[:, 1] >= threshold_low
filtered_data_points = filtered_data_points[lpf_mask]

# remove top 5%
filtered_data_points = np.array(filtered_data_points)
threshold_high = highest_y - (highest_y / 20)
mask = filtered_data_points[:, 1] < threshold_high
filtered_data_points = filtered_data_points[mask]

#sort by lowest x
filtered_data_points = np.array(filtered_data_points)
sorted_data_points = filtered_data_points[np.argsort(filtered_data_points[:, 0])]

#interpolate
sorted_data_points = np.array(sorted_data_points)
x_coords = sorted_data_points[:, 0]
y_coords = sorted_data_points[:, 1]
interp_func = interp1d(y_coords, x_coords)
desired_y_values = np.arange(min(y_coords), max(y_coords), 5)
interpolated_x_values = interp_func(desired_y_values)
interpolated_points = np.column_stack((interpolated_x_values, desired_y_values))

#flip graph upside down
interpolated_points_flipped = np.array(interpolated_points)
highest_y = np.max(filtered_data_points[:, 0])
print(highest_y)
for i in range(len(interpolated_points)):
    interpolated_points_flipped[i][0] = highest_y - interpolated_points_flipped[i][0]

#flip graph upside down
interpolated_points_flipped = np.array(interpolated_points)
highest_y = np.max(filtered_data_points[:, 0])
print(highest_y)
for i in range(len(interpolated_points)):
    interpolated_points_flipped[i][0] = highest_y - interpolated_points_flipped[i][0]



#scale image
interpolated_points_flipped_and_scaled = np.array(interpolated_points_flipped)
#define scale ratio
highest_y = np.max(interpolated_points_flipped_and_scaled[:, 0])
highest_x = np.max(interpolated_points_flipped_and_scaled[:, 1])

y_ratio = ((POT_WIDTH_CM/2) / highest_y) * CM_TO_MM
x_ratio = ((POT_HEIGHT_CM) / highest_x) * CM_TO_MM #lying on its side.

print("Highest x:y:", highest_x, highest_y)
scaled_data_points_mm = []
for point in interpolated_points_flipped_and_scaled:
    scaled_point = []
    if point[0] != 0:
        scaled_point.append(int(point[0] * y_ratio))
    if point[1] != 0:
        scaled_point.append(int(point[1] * x_ratio))
    scaled_data_points_mm.append(scaled_point)


#plot
scaled_data_points_mm = np.array(scaled_data_points_mm)
#define scale ratio
highest_y_mm = np.max(scaled_data_points_mm[:, 0])
highest_x_mm = np.max(scaled_data_points_mm[:, 1])

x_coords_mm = scaled_data_points_mm[:, 1]
y_coords_mm = scaled_data_points_mm[:, 0]
x_min = 0  # Minimum x-value of the range
x_max = highest_x_mm  # Maximum x-value of the range


degree = 10  # Degree of the polynomial
coefficients = np.polyfit(x_coords_mm, y_coords_mm, degree)
polynomial = np.poly1d(coefficients)


x_values = np.linspace(min(x_coords_mm), max(x_coords_mm), 1000)

# Evaluate the polynomial at the x-values
y_values = polynomial(x_values)

# Calculate the area under the curve using the trapezoidal rule
area = trapz(y_values, x_values)
area_cm2 = area / 100
print("Area under the curve:", area_cm2)

def integrand(x):
    return np.abs(polynomial(x))

volume, error = quad(integrand, 0, highest_x_mm)
print(highest_x_mm)
print("Volume of the cone:", volume)

x = np.linspace(0, highest_x_mm, 2000)
# y = np.polyval(coefficients, x)



# Plot the data points
plt.scatter(x_coords_mm, y_coords_mm, c='blue', label='Data Points')

# plt.plot(x, y)
# Set plot title and labels
plt.title('Scatter Plot with Polynomial Curve')
plt.xlabel('X')
plt.ylabel('Y')

# Add legend
plt.legend()

# Show the plot
plt.show()