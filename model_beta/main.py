import cv2
import os
import torch as pt
import numpy as np
import tkinter as tk
from tkinter import filedialog

from yunet import YuNet
from sface import SFace



# Hàm chính để tích hợp phát hiện và nhận diện
if __name__ == '__main__':


    # Định nghĩa các thư mục
    # Input_dir là đường dẫn đến thư mục chứa các hình ảnh đầu vào, tức là các khuôn mặt đã được phát hiện và cắt từ video hoặc nguồn hình ảnh khác.

    project_dir = r'model_beta\\'

    input_dir = r'model_beta\\input_folder'

    input_image_dir = r"model_beta\\input_image"

    output_dir = r'model_beta\\output_folder'

    detected_frame_dir = r'model_beta\\detected_frame_folder'

    detect_model_path = r"model_train\\yolov8n-face.pt"  # Thay thế bằng đường dẫn thực tế đến mô hình YOLOv8 của bạn





    # Nhập ảnh mục tiêu
    target_img_path = import_Image(input_image_dir)
    if not target_img_path:
        print("Target image not set. Exiting.")
        exit(1)

    # Khởi tạo mô hình phát hiện
    detect_model_instance = detect_Model(detect_model_path, device="cpu")

    # Khởi tạo các mô hình nhận diện
    face_detector = YuNet(modelPath= r'model_train\\yunet.onnx',
                          inputSize=[320, 320],
                          confThreshold=0.8,
                          nmsThreshold=0.3,
                          topK=5000,
                          backendId=cv2.dnn.DNN_BACKEND_OPENCV,
                          targetId=cv2.dnn.DNN_TARGET_CPU)

    face_recognizer = SFace(modelPath= r'model_train\\reg.onnx',
                            disType=0,
                            backendId=cv2.dnn.DNN_BACKEND_OPENCV,
                            targetId=cv2.dnn.DNN_TARGET_CPU)

    # Đảm bảo các thư mục đầu vào và đầu ra tồn tại
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ví dụ: Mở webcam và thực hiện phát hiện
    video_source = 0 # r"video_test\video1.mp4"  # 0 cho webcam mặc định, hoặc cung cấp đường dẫn tới tệp video
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit(1)

    count_video_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Thực hiện phát hiện trên khung hình
        detected_frame = detect_Frame(detect_model_instance, frame, input_dir, detected_frame_dir, count_video_frame)

        count_video_frame += 1

        # Hiển thị khung hình đã phát hiện
        cv2.imshow('Detected Frame', detected_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Thực hiện quá trình nhận diện
    process_Images(input_dir, target_img_path, face_detector, face_recognizer, output_dir)





