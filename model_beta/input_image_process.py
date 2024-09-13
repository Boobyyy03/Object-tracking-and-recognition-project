import os
import tkinter as tk
import cv2
from tkinter import filedialog




def import_Image(input_image_dir):
    # Tạo cửa sổ Tkinter
    root = tk.Tk()
    root.withdraw()

    # Yêu cầu người dùng chọn tệp ảnh
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])

    if not file_path:
        print("No file selected!")
        return None

    # Yêu cầu nhập tên của người trong ảnh
    person_name = input("Enter the person's name: ")

    # Đọc tệp ảnh
    image = cv2.imread(file_path)

    # Định nghĩa thư mục lưu ảnh đầu vào
    if not os.path.exists(input_image_dir):
        os.makedirs(input_image_dir)


    # Lưu ảnh với tên người làm tên tệp
    if image is not None:
        image_path = os.path.join(input_image_dir, f"{person_name}.jpg")
        cv2.imwrite(image_path, image)
        print("Image saved successfully!")
        return image_path
    else:
        print("Failed to load image!")
        return None


