# lấy hình ảnh từ folder "input" so sánh với file target trả ra folder "output\(id)". Mỗi người nhận diện được để ở folder riêng.


import cv2 as cv
import numpy as np
import os
from yunet import YuNet
from sface import SFace


def process_images(input_dir, target_img_path, model_face_detector, model_face_recognizer, output_dir):
    target_img = cv.imread(target_img_path)
    if target_img is None:
        print(f"Error: Could not load target face image from {target_img_path}")
        return

    model_face_detector.setInputSize([target_img.shape[1], target_img.shape[0]])
    target_faces = model_face_detector.infer(target_img)
    if target_faces.shape[0] == 0:
        print("Error: No face detected in target image.")
        return

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv.imread(img_path)

        if img is None:
            print(f"Warning: Could not load image {img_name}")
            continue

        parts = img_name.split('_')
        if len(parts) < 3:
            print(f"Error: Invalid filename format for {img_name}. Skipping.")
            continue
        img_id = parts[1]
        frame_num = parts[2].split('.')[0]

        print(f"Processing {img_name}...")

        model_face_detector.setInputSize([img.shape[1], img.shape[0]])

        detected_faces = model_face_detector.infer(img)

        matches = []
        scores = []

        for face in detected_faces:
            score, match = model_face_recognizer.match(target_img, target_faces[0][:-1], img, face[:-1])
            matches.append(match)
            scores.append(score)

        img = visualize_recognition(img, detected_faces, target_img, matches, scores)

        output_subdir = os.path.join(output_dir, img_id)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        output_img_path = os.path.join(output_subdir, f"output_{img_name}")
        cv.imwrite(output_img_path, img)

    print(f"All images processed. Output saved to {output_dir}")


def visualize_recognition(frame, faces, target_img, matches, scores, box_color=(0, 255, 0), text_color=(255, 255, 255)):
    for i, face in enumerate(faces):
        x, y, w, h = face[:4].astype(np.int32)
        box_color = (0, 255, 0) if matches[i] else (0, 0, 255)
        cv.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

        score_text = f'Score: {scores[i]:.2f}'
        cv.putText(frame, score_text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) 

    return frame


if __name__ == '__main__':
    face_detector = YuNet(modelPath='yunet.onnx',
                          inputSize=[320, 320],
                          confThreshold=0.8,
                          nmsThreshold=0.3,
                          topK=5000,
                          backendId=cv.dnn.DNN_BACKEND_OPENCV,
                          targetId=cv.dnn.DNN_TARGET_CPU)

    face_recognizer = SFace(modelPath='reg.onnx',
                            disType=0,
                            backendId=cv.dnn.DNN_BACKEND_OPENCV,
                            targetId=cv.dnn.DNN_TARGET_CPU)

    input_dir = 'input'
    output_dir = 'output'
    target_img_path = 'img5.jpg'

    process_images(input_dir, target_img_path, face_detector, face_recognizer, output_dir)
