import cv2
import os
import matplotlib.pyplot as plt
import torch as pt
from ultralytics import YOLO




def detect_model(link_detect_model, device = "cpu"):

    # Choose device
    device = pt.device(device)

    # Load the YOLOv8 model
    model = YOLO(link_detect_model, task='track')
    model.to(device=device)

    return model




def detect(detect_model, frame, link_output_folder, count_video_frame, conf_threshold = 0.5):

    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = detect_model.track(frame, persist=True)
    
    # Take bounding boxes and infomation
    for detect_object in results[0].boxes:
        id, co, bb = detect_object.id, detect_object.conf, detect_object.data[0,:4]
        x1, y1, x2, y2 = map(int, bb)
        try:
            id = int(id)
        except:
            id = 0

        if co < conf_threshold:
            continue

        # Crop face and save in output folder
        image_face = frame[y1:y2, x1:x2]
        cv2.imwrite(os.linkdir(link_output_folder, str(id) + "_" + str(count_video_frame) + ".png"), image_face)

        # Draw bounding boxes on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + 20, y1), (0, 0, 255), -1)
        cv2.putText(frame, str(id), (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
    return frame