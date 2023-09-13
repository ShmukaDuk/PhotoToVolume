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
image_path = ""

def select_image():
    global image_path  # Declare image_path as global
    image_path = filedialog.askopenfilename()
    print("Selected image:", image_path)

def runCalculation(image_path):
    print("Grinding image:", image_path)

    calculator = CalculateVolume(image_path)
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

# Create a button to generate
generate_button = tk.Button(frame, text="Generate", command=lambda: runCalculation(image_path))
generate_button.pack()

# Start the Tkinter main loop
root.mainloop()
