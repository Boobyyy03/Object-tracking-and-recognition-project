import os
import tkinter as tk
import cv2
from tkinter import filedialog
def import_image():
    # Create a Tkinter window
    root = tk.Tk()
    root.withdraw()

    # Ask the user to select an image file
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])

    # Ask for the person's name
    person_name = input("Enter the person's name: ")

    # Open the image file
    image = cv2.imread(file_path)

    # Check if there is an image in the input folder
    if os.path.exists("model_beta/input_image"):
        # Remove the existing image
        existing_image = os.listdir("model_beta/input_image")
        if len(existing_image) > 0:
            existing_image_path = os.path.join("model_beta/input_image", existing_image[0])
            if os.path.isfile(existing_image_path):
                os.remove(existing_image_path)
            
    # # Create a folder for the person if it doesn't exist
    # person_folder = os.path.join("model_beta/input_image", person_name)
    # # if not os.path.exists(person_folder):
    # #     os.makedirs(person_folder)

    # Save the image with the person's name as the file name
    if image is not None:
        image_path = os.path.join('model_beta/input_image',f"{person_name}.jpg")
        cv2.imwrite(image_path, image)
        print("Image saved successfully!")
    else:
        print("Failed to load image!")


