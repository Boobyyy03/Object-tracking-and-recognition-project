import cv2
import os
import torch as pt
from ultralytics import YOLO
import numpy as np
from yunet import YuNet
from sface import SFace
import tkinter as tk
from tkinter import filedialog
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Hàm khởi tạo mô hình phát hiện
def detect_model(link_detect_model, device="cpu"):
    device = pt.device(device)
    model = YOLO(link_detect_model, task='track')
    model.to(device=device)
    return model

# Hàm phát hiện đối tượng trong khung hình
def detect(detect_model, frame, link_output_folder, count_video_frame, conf_threshold=0.5):
    results = detect_model.track(frame, persist=True)

    for detect_object in results[0].boxes:
        id, co, bb = detect_object.id, detect_object.conf, detect_object.data[0,:4]
        x1, y1, x2, y2 = map(int, bb)
        try:
            id = int(id)
        except:
            id = 0

        if co < conf_threshold:
            continue

        image_face = frame[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(link_output_folder, f"{id}_{count_video_frame}.jpg"), image_face)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + 20, y1), (0, 0, 255), -1)
        cv2.putText(frame, str(id), (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    return frame

# Hàm xử lý nhận diện khuôn mặt
def process_images(input_dir, target_img_path, model_face_detector, model_face_recognizer, output_dir):
    target_img = cv2.imread(target_img_path)
    if target_img is None:
        print(f"Error: Could not load target face image from {target_img_path}")
        return

    model_face_detector.setInputSize([target_img.shape[1], target_img.shape[0]])
    target_faces = model_face_detector.infer(target_img)
    if target_faces.shape[0] == 0:
        print("Error: No face detected in target image.")
        return

    processed_count = 0  # Counter to keep track of processed images
    max_images_to_process = 3  # Limit the number of images to process

    for img_name in os.listdir(input_dir):
        if processed_count >= max_images_to_process:
            break  # Stop processing once 3 images have been processed

        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Could not load image {img_name}")
            continue

        parts = img_name.split('_')
        if len(parts) < 2:
            print(f"Error: Invalid filename format for {img_name}. Skipping.")
            continue
        img_id = parts[0]
        frame_num = parts[1].split('.')[0]

        print(f"Processing {img_name}...")

        model_face_detector.setInputSize([img.shape[1], img.shape[0]])
        detected_faces = model_face_detector.infer(img)

        sufficient_landmarks_found = False

        for face in detected_faces:
            if detect_face_landmarks(img, face):  # Check for sufficient landmarks
                sufficient_landmarks_found = True
                score, match = model_face_recognizer.match(target_img, target_faces[0][:-1], img, face[:-1])

                img = visualize_recognition(img, detected_faces, target_img, [match], [score])

                output_subdir = os.path.join(output_dir, img_id)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                output_img_path = os.path.join(output_subdir, f"output_{img_name}")
                cv2.imwrite(output_img_path, img)
                processed_count += 1  # Increment the processed image count
                break  # Only process the first face in the image

        if not sufficient_landmarks_found:
            print(f"Skipping {img_name} due to insufficient landmarks.")

    print(f"Processed {processed_count} images. Output saved to {output_dir}")


def detect_face_landmarks(img, face):
    """
    Detect facial landmarks using Mediapipe Face Mesh and ensure sufficient landmarks are detected.
    Returns True if sufficient landmarks are found, otherwise False.
    """
    # Extract face bounding box coordinates
    x, y, w, h = face[:4].astype(np.int32)

    # Ensure bounding box is within image bounds
    h_img, w_img, _ = img.shape
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)

    
    cropped_face = img[y:y + h, x:x + w]

    if cropped_face.size == 0:
        print("Warning: Cropped face is empty, skipping this face.")
        return False

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        result = face_mesh.process(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
        if result.multi_face_landmarks:
            for landmark in result.multi_face_landmarks:
                return True
    return False


def visualize_recognition(frame, faces, target_img, matches, scores, box_color=(0, 255, 0), text_color=(255, 255, 255)):
    for i, face in enumerate(faces):
        x, y, w, h = face[:4].astype(np.int32)
        box_color = (0, 255, 0) if matches[i] else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

        score_text = f'Score: {scores[i]:.2f}'
        font_scale = 0.6
        font_thickness = 2
        text_size, _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_w, text_h = text_size

        text_x = x + 5
        text_y = y + text_h + 5

        if text_y + text_h > y + h:
            text_y = y + h - 5

        cv2.putText(frame, score_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    return frame


# Hàm nhập ảnh mục tiêu
def import_image():
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
    input_image_dir = "model_beta/input_image"
    if not os.path.exists(input_image_dir):
        os.makedirs(input_image_dir)

    # Xóa các ảnh hiện có trong thư mục đầu vào


    # Lưu ảnh với tên người làm tên tệp
    if image is not None:
        image_path = os.path.join(input_image_dir, f"{person_name}.jpg")
        cv2.imwrite(image_path, image)
        print("Image saved successfully!")
        return image_path
    else:
        print("Failed to load image!")
        return None

# Hàm chính để tích hợp phát hiện và nhận diện
if __name__ == '__main__':
    # Nhập ảnh mục tiêu
    target_img_path = import_image()
    if not target_img_path:
        print("Target image not set. Exiting.")
        exit(1)

    # Khởi tạo mô hình phát hiện
    detect_model_path = r"model_train\yolov8n-face.pt"  # Thay thế bằng đường dẫn thực tế đến mô hình YOLOv8 của bạn
    detect_model_instance = detect_model(detect_model_path, device="cpu")

    # Khởi tạo các mô hình nhận diện
    face_detector = YuNet(modelPath= r'model_train\yunet.onnx',
                          inputSize=[320, 320],
                          confThreshold=0.8,
                          nmsThreshold=0.3,
                          topK=5000,
                          backendId=cv2.dnn.DNN_BACKEND_OPENCV,
                          targetId=cv2.dnn.DNN_TARGET_CPU)

    face_recognizer = SFace(modelPath= r'model_train\reg.onnx',
                            disType=0,
                            backendId=cv2.dnn.DNN_BACKEND_OPENCV,
                            targetId=cv2.dnn.DNN_TARGET_CPU)

    # Định nghĩa các thư mục
    # Input_dir là đường dẫn đến thư mục chứa các hình ảnh đầu vào, tức là các khuôn mặt đã được phát hiện và cắt từ video hoặc nguồn hình ảnh khác.
    input_dir = r'model_beta\input_folder'

    output_dir = r'model_beta\output_folder'

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

        count_video_frame += 1

        # Thực hiện phát hiện trên khung hình
        detected_frame = detect(detect_model_instance, frame, input_dir, count_video_frame)

        # Hiển thị khung hình đã phát hiện
        cv2.imshow('Detected Frame', detected_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Thực hiện quá trình nhận diện
    process_images(input_dir, target_img_path, face_detector, face_recognizer, output_dir)
