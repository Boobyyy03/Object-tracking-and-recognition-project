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


def get_Date_Time(decimal_sec = 1):
    # Get date time in format to save
    datetime_frame = str(datetime.datetime.now()).split(" ")
    date_frame = datetime_frame[0].split("-")
    time_frame = datetime_frame[1].split(":")

    time_frame.append(time_frame[2].split(".")[1][:decimal_sec])
    time_frame[2] = time_frame[2].split(".")[0]

    return " " + " ".join(date_frame) + " ".join(time_frame)

def cos_similarity(a, b):
    return pt.cos(pt.sum(a*b)/(pt.sum(a**2)**0.5 * pt.sum(b**2)**0.5))


def detect_Frame(detect_model, frame, dict_id_image, count_dict, resnet, link_output_folder, link_detected_frame_folder, camera, count_video_frame,
                 conf_threshold=0.5):
    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = detect_model.track(frame, persist=True)
    shape_face_now = 120
    list_new_info = []
    list_new_image = []


    # Take bounding boxes and infomation
    for detect_object in results[0].boxes:
        id, co, bb = detect_object.id, detect_object.conf, detect_object.data[0, :4]
        x1, y1, x2, y2 = map(int, bb)
        face_now = cv2.resize(frame[int(y1):int(y2), int(x1):int(x2)], (shape_face_now, shape_face_now))
            
        try:
            id = int(id)
        except:
            id = 0
        
        id_show = id
                
        if len(dict_id_image) < 2:
            dict_id_image[id] = [[id], face_now, face_now, face_now, face_now]
        else:
            add_dict = 0
            for case in range(1, len(dict_id_image)):
                #print(dict_id_image[case][0])
                if id in dict_id_image[case][0]:
                    dict_id_image[case][2:4] = dict_id_image[case][3:]
                    dict_id_image[case][4] = face_now
                    id_show = case
                    add_dict = 1
                    break
                else:
                    if count_dict < id:
                        count_dict = id
                        image_pt_1 = pt.zeros((1, 3, shape_face_now, shape_face_now))
                        image_pt_2 = pt.zeros((1, 3, shape_face_now, shape_face_now))
                        image_pt_3 = pt.zeros((1, 3, shape_face_now, shape_face_now))
                        image_pt_4 = pt.zeros((1, 3, shape_face_now, shape_face_now))
                        image_pt = pt.zeros((1, 3, shape_face_now, shape_face_now))

                        for i in range(3):
                            image_pt_1[0, i] = pt.tensor(dict_id_image[case][1][:,:,2-i])
                            image_pt_2[0, i] = pt.tensor(dict_id_image[case][2][:,:,2-i])
                            image_pt_3[0, i] = pt.tensor(dict_id_image[case][3][:,:,2-i])
                            image_pt_4[0, i] = pt.tensor(dict_id_image[case][4][:,:,2-i])
                            image_pt[0, i] = pt.tensor(face_now[:,:,2-i])

                        image_pt_1 = resnet(image_pt_1)
                        image_pt_2 = resnet(image_pt_2)
                        image_pt_3 = resnet(image_pt_3)
                        image_pt_4 = resnet(image_pt_4)
                        image_pt = resnet(image_pt)
                        
                        mean_similarity = (cos_similarity(image_pt, image_pt_1) + cos_similarity(image_pt, image_pt_2) + cos_similarity(image_pt, image_pt_3) + cos_similarity(image_pt, image_pt_4))/4

                        if mean_similarity > 0.93:
                            dict_id_image[case][0].append(id)
                            dict_id_image[case][2:4] = dict_id_image[case][3:]
                            dict_id_image[case][4] = face_now
                            id_show = case
                            add_dict = 1
                            break
                        else:
                            continue
            if add_dict == 0:
                id_show = len(dict_id_image) + 1
                dict_id_image[len(dict_id_image) + 1] = [[id], face_now, face_now]

        if co < conf_threshold:
            continue

        # Get time now
        datetime_frame = get_Date_Time()


        # Crop face and save in output folder
        #image_face = frame[y1:y2, x1:x2]
        #cv2.imwrite(os.path.join(link_output_folder, f"{camera}_{id_show}_{count_video_frame}{datetime_frame}.png"), image_face)
        list_new_info.append(f"{camera} {id_show} {count_video_frame}{datetime_frame}")
        list_new_image.append(transforms.ToTensor(face_now))

        # Draw bounding boxes on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + 20, y1), (0, 0, 255), -1)
        cv2.putText(frame, str(id_show), (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        

    # Add info and face to file
    fileo = open(os.path.join(self.input_dir, str(camera) + ".txt"), "a")
    for info in list_new_info:
        fileo.write(info + "\n")
    fileo.close()

    tensor_list = pt.zeros((len(list_new_image), shape_face_now, shape_face_now, 3))
    for count in range(len(list_new_image)):
        tensor_list[count, :, :, :] = transforms.ToTensor(list_new_image[count])

    try:
        tensor_open = pt.load(os.path.join(self.input_dir, str(camera) + ".pt"))
        len_tensor = tensor_open.shape[0] + len(list_new_image)
        tensor_new = pt.zeros((len_tensor, shape_face_now, shape_face_now, 3))
        tensor_new[:tensor_open.shape[0], :, :, :] = tensor_open[:, :, :, :]
        tensor_new[tensor_open.shape[0]:, :, :, :] = tensor_list[:, :, :, :]
        pt.save(tensor_new, os.path.join(self.input_dir, str(camera) + ".pt"))
    except
        pt.save(tensor_list, os.path.join(self.input_dir, str(camera) + ".pt"))

    
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

    return frame, dict_id_image, count_dict
