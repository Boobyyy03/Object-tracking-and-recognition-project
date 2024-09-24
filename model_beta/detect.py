import cv2
import os
import matplotlib.pyplot as plt
import torch as pt
import datetime
from ultralytics import YOLO



def detect_Model(link_detect_model, device="cpu"):
    # Choose device
    device = pt.device(device)

    # Load the YOLOv8 model
    model = YOLO(link_detect_model, task='track')
    model.to(device=device)

    return model


def get_Date_Time(decimal_sec = 1):
    # Get date time in format to save
    datetime_frame = str(datetime.datetime.now()).split(" ")
    date_frame = datetime_frame[0].split("-")
    time_frame = datetime_frame[1].split(":")

    time_frame.append(time_frame[2].split(".")[1][:decimal_sec])
    time_frame[2] = time_frame[2].split(".")[0]

    return "_" + "".join(date_frame) + "".join(time_frame)


def detect_Frame(detect_model, frame, link_output_folder, link_detected_frame_folder, camera, count_video_frame,
                 conf_threshold=0.5):
    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = detect_model.track(frame, persist=True)

    # Take bounding boxes and infomation
    for detect_object in results[0].boxes:
        id, co, bb = detect_object.id, detect_object.conf, detect_object.data[0, :4]
        x1, y1, x2, y2 = map(int, bb)
        try:
            id = int(id)
        except:
            id = 0

        if co < conf_threshold:
            continue

        # Get time now
        datetime_frame = get_Date_Time()


        # Crop face and save in output folder
        image_face = frame[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(link_output_folder, f"{camera}_{id}_{count_video_frame}{datetime_frame}.jpg"), image_face)

        # Draw bounding boxes on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + 20, y1), (0, 0, 255), -1)
        cv2.putText(frame, str(id), (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
    
    # Create name for tracked image (frame)
    name_frame = ""
    for i in range(1,7):
        if count_video_frame < 10**i:
            for ii in range(6-i):
                name_frame = name_frame + "0"
            break
        else:
            continue
    datetime_frame = get_Date_Time()
    name_frame = os.path.join(str(camera), name_frame + str(count_video_frame) + datetime_frame + ".png")

    # Lưu hình ảnh vào địa chỉ
    cv2.imwrite(os.path.join(link_detected_frame_folder , name_frame), frame)

    return frame
