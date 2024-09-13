# lấy hình ảnh từ folder "input" so sánh với file target trả ra folder "output\(id)". Mỗi người nhận diện được để ở folder riêng.


import cv2
import numpy as np
import os
from yunet import YuNet
from sface import SFace
import mediapipe as mp

import shutil

import shutil

def process_Images(input_dir, target_img_path, model_face_detector, model_face_recognizer, output_dir):
    target_img = cv2.imread(target_img_path)
    if target_img is None:
        print(f"Error: Could not load target face image from {target_img_path}")
        return

    target_img_name = os.path.splitext(os.path.basename(target_img_path))[0]

    model_face_detector.setInputSize([target_img.shape[1], target_img.shape[0]])
    target_faces = model_face_detector.infer(target_img)
    if target_faces.shape[0] == 0:
        print("Error: No face detected in target image.")
        return

    processed_count_per_id = {}
    max_images_to_process = 3
    folders_to_rename = {}

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Could not load image {img_name}")
            continue

        parts = img_name.split('_')
        if len(parts) != 3:
            print(f"Error: Invalid filename format for {img_name}. Skipping.")
            continue
        cam_id = parts[0]  # Camera ID
        img_id = parts[1]  # Person ID
        frame_num = parts[2].split('.')[0]  # Frame number, without the file extension

        # Ensure the processed count per cam_id and img_id
        if cam_id not in processed_count_per_id:
            processed_count_per_id[cam_id] = {}
        if img_id not in processed_count_per_id[cam_id]:
            processed_count_per_id[cam_id][img_id] = 0

        if processed_count_per_id[cam_id][img_id] >= max_images_to_process:
            continue

        print(f"Processing {img_name}...")

        model_face_detector.setInputSize([img.shape[1], img.shape[0]])
        detected_faces = model_face_detector.infer(img)

        # Initialize Mediapipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh

        sufficient_landmarks_found = False

        for face in detected_faces:
            if detect_Face_Landmarks(img, face, mp_face_mesh):
                sufficient_landmarks_found = True
                score, match = model_face_recognizer.match(target_img, target_faces[0][:-1], img, face[:-1])

                cam_output_subdir = os.path.join(output_dir, cam_id)
                img_output_subdir = os.path.join(cam_output_subdir, img_id)

                if not os.path.exists(img_output_subdir):
                    os.makedirs(img_output_subdir)

                img = visualize_Recognition(img, detected_faces, target_img, [match], [score])

                output_img_path = os.path.join(img_output_subdir, f"output_{img_name}")
                cv2.imwrite(output_img_path, img)

                processed_count_per_id[cam_id][img_id] += 1

                if match:
                    folders_to_rename[img_output_subdir] = os.path.join(cam_output_subdir, target_img_name)

                break

        if not sufficient_landmarks_found:
            print(f"Skipping {img_name} due to insufficient landmarks.")

    for original_folder, new_folder in folders_to_rename.items():
        if not os.path.exists(new_folder):  # Ensure target folder doesn't already exist
            shutil.move(original_folder, new_folder)
        else:
            print(f"Warning: Target folder '{new_folder}' already exists. Skipping rename for '{original_folder}'.")

    print(f"Processing complete. Output saved to {output_dir}")


def detect_Face_Landmarks(img, face, mp_face_mesh):
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


def visualize_Recognition(frame, faces, target_img, matches, scores, box_color=(0, 255, 0), text_color=(0, 0, 255)):
    for i, face in enumerate(faces):
        x, y, w, h = face[:4].astype(np.int32)
        box_color = (0, 255, 0) if matches[i] else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

        score_text = f'{scores[i]:.2f}'

        font_scale = h / 150.0
        font_thickness = max(1, int(h / 50))

        text_size, _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_w, text_h = text_size

        text_x = x + 5
        text_y = y + text_h + 5

        if text_y + text_h > y + h:
            text_y = y + h - 5

        cv2.putText(frame, score_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
                    font_thickness)

    return frame
