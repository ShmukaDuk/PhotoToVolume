import tkinter as tk
from tkinter import filedialog
import sys
import os

# Get the parent directory of the current file (test.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

from src.photo_to_volume import CalculateVolume

global image_path
global pot_width
global pot_height

def update_pot_height(value):
    global pot_height
    pot_height = float(value)

def update_pot_width(value):
    global pot_width
    pot_width = float(value)

def select_image():
    global image_path  # Declare image_path as global
    image_path = filedialog.askopenfilename()
    print("Selected image:", image_path)

def runCalculation():
    print("Grinding image:", image_path)

    calculator = CalculateVolume(image_path, pot_width, pot_height, 50)
    calculator.prep_image()
    volume_litres = calculator.calculate_volume()
    print("Volume in Liters:", volume_litres)

# Create the main application window
root = tk.Tk()
root.title("Image Location Generator")

# Create a frame for the GUI elements
frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

# Create a label for image selection
image_path_label = tk.Label(frame, text="Select an image:")
image_path_label.pack()

# Create a button to select an image
select_button = tk.Button(frame, text="Select Image", command=select_image)
select_button.pack()

# Create sliders to set POT_HEIGHT_CM and POT_WIDTH_CM
height_label = tk.Label(frame, text="Pot Height (cm):")
height_label.pack()
height_slider = tk.Scale(frame, from_=0, to=100, orient="horizontal", length=200, resolution=0.1, command=update_pot_height)
height_slider.pack()

width_label = tk.Label(frame, text="Pot Width (cm):")
width_label.pack()
width_slider = tk.Scale(frame, from_=0, to=100, orient="horizontal", length=200, resolution=0.1, command=update_pot_width)
width_slider.pack()

# Create a button to generate
generate_button = tk.Button(frame, text="Generate", command=runCalculation)
generate_button.pack()

# Start the Tkinter main loop
root.mainloop()
