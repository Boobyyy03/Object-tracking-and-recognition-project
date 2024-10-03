import cv2
import os
import matplotlib.pyplot as plt
import torch as pt
import datetime
import torchvision.transforms as transforms
from ultralytics import YOLO


def detect_Model(link_detect_model, device="cpu"):
    # Choose device
    device = pt.device(device)

    # Load the YOLOv8 model
    model = YOLO(link_detect_model, task='track')
    model.to(device=device)

    return model


def get_Date_Time(decimal_sec=0):
    # Get date time in format to save
    datetime_frame = str(datetime.datetime.now()).split(" ")
    date_frame = datetime_frame[0].split("-")
    time_frame = datetime_frame[1].split(":")

    time_frame.append(time_frame[2].split(".")[1][:decimal_sec])
    time_frame[2] = time_frame[2].split(".")[0]

    return " " + " ".join(date_frame) + " " + " ".join(time_frame), "_" + "_".join(date_frame) + "_".join(time_frame)


def cos_similarity(a, b):
    cos_cal = pt.nn.CosineSimilarity(dim=1, eps=1e-10)
    return cos_cal(a, b)


def detect_Frame(detect_model, frame, dict_id_image, count_dict, resnet, link_output_folder, link_detected_frame_folder,
                 camera, count_video_frame,
                 conf_threshold=0.5):
    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = detect_model.track(frame, persist=True)
    shape_face_now = [100, 75]
    list_new_info = []
    list_new_image = []

    transform = transforms.ToTensor()

    # Take bounding boxes and infomation
    for detect_object in results[0].boxes:
        id, co, bb = detect_object.id, detect_object.conf, detect_object.data[0, :4]
        x1, y1, x2, y2 = map(int, bb)
        face_now = cv2.resize(frame[int(y1):int(y2), int(x1):int(x2)], (shape_face_now[1], shape_face_now[0]))

        try:
            id = int(id)
        except:
            id = 0

        id_show = id

        if co < conf_threshold:
            continue

        # Get time now
        datetime_frame1, datetime_frame2 = get_Date_Time()

        # Crop face and save in output folder
        # image_face = frame[y1:y2, x1:x2]
        # cv2.imwrite(os.path.join(link_output_folder, f"{camera}_{id_show}_{count_video_frame}{datetime_frame}.png"), image_face)
        list_new_info.append(f"{camera} {id_show} {count_video_frame}{datetime_frame1}")
        list_new_image.append(transform(face_now))

        # Draw bounding boxes on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + 20, y1), (0, 0, 255), -1)
        cv2.putText(frame, str(id_show), (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Add info and face to file
    fileo = open(os.path.join(link_output_folder, str(camera) + ".txt"), "a")
    for info in list_new_info:
        fileo.write(info + "\n")
    fileo.close()

    tensor_list = pt.zeros((len(list_new_image), 3, shape_face_now[0], shape_face_now[1]))
    for count in range(len(list_new_image)):
        tensor_list[count, :, :, :] = list_new_image[count]

    try:
        tensor_open = pt.load(os.path.join(link_output_folder, str(camera) + ".pt"))
        len_tensor = tensor_open.shape[0] + len(list_new_image)
        tensor_new = pt.zeros((len_tensor, 3, shape_face_now[0], shape_face_now[1]))
        tensor_new[:tensor_open.shape[0], :, :, :] = tensor_open[:, :, :, :]
        tensor_new[tensor_open.shape[0]:, :, :, :] = tensor_list[:, :, :, :]
        pt.save(tensor_new, os.path.join(link_output_folder, str(camera) + ".pt"))
    except:
        pt.save(tensor_list, os.path.join(link_output_folder, str(camera) + ".pt"))

    # Create name for tracked image (frame)
    name_frame = ""
    for i in range(1, 7):
        if count_video_frame < 10 ** i:
            for ii in range(6 - i):
                name_frame = name_frame + "0"
            break
        else:
            continue
    _, datetime_frame2 = get_Date_Time()
    name_frame = os.path.join(str(camera), name_frame + str(count_video_frame) + datetime_frame2 + ".png")

    # Lưu hình ảnh vào địa chỉ
    cv2.imwrite(os.path.join(link_detected_frame_folder, name_frame), frame)

    return frame, dict_id_image, count_dict
