import cv2
import numpy as np
import os
import torch as pt
from yunet import YuNet
from sface import SFace
from facenet_pytorch import MTCNN
import shutil
import torch as pt
from torchvision import transforms



def cos_similarity(a, b):
    cos_cal = pt.nn.CosineSimilarity(dim=1, eps=1e-10)
    return cos_cal(a, b)

def process_Images(number_camera, resnet, input_dir, target_img_path, model_face_detector, output_dir):
    target_img = cv2.imread(target_img_path)
    shape_face = [100, 75]
    transform = transforms.ToTensor()
    file_open = open(os.path.join(output_dir, "info_similar.txt"), "w")


    if target_img is None:
        print(f"Error: Could not load target face image from {target_img_path}")
        return

    target_img_name = os.path.splitext(os.path.basename(target_img_path))[0]

    model_face_detector.setInputSize([target_img.shape[1], target_img.shape[0]])
    target_faces = model_face_detector.infer(target_img)
    if target_faces.shape[0] == 0:
        print("Error: No face detected in target image.")
        return
    else:
        x, y, w, h = target_faces[0][:4].astype(np.int32)
        target_img = target_img[y:y + h, x:x + w,:]
        cv2.imwrite(os.path.join(output_dir, "keanu" + ".png"), target_img)


    #processed_count_per_id = {}
    #score_per_id = {}  # Dictionary to store scores per img_id
    #max_images_to_process = 3
    #folders_to_rename = {}

    for i in range(number_camera):

        tensor_open = pt.load(os.path.join(input_dir, str(i) + ".pt"))
        tensor_open_now = tensor_open.to(device="cuda")
        with open(os.path.join(input_dir, str(i) + ".txt")) as i_o:
            info_open = [line.rstrip() for line in i_o]


        feature_faces = resnet(tensor_open_now)

        feature_target = transform(cv2.resize(target_img, (shape_face[1], shape_face[0])))
        feature_target = feature_target[None, :, :, :]
        feature_target = feature_target.to(device="cuda")
        feature_target = resnet(feature_target)

        cos_values = cos_similarity(feature_faces, feature_target)
        cos_values = cos_values.to(device="cpu")

        #img_path = os.path.join(input_dir, img_name)
        #img = cv2.imread(img_path)

        #if img is None:
        #    print(f"Warning: Could not load image {img_name}")
        #    continue

        #parts = img_name.split('_')
        #if len(parts) != 3:
        #    print(f"Error: Invalid filename format for {img_name}. Skipping.")
        #    continue
        #cam_id = parts[0]  # Camera ID
        #img_id = parts[1]  # Person ID
        #frame_num = parts[2].split('.')[0]  # Frame number, without the file extension
        #print(parts)

        # Ensure the processed count per cam_id and img_id
        #if cam_id not in processed_count_per_id:
         #   processed_count_per_id[cam_id] = {}
       # if img_id not in processed_count_per_id[cam_id]:
          #  processed_count_per_id[cam_id][img_id] = 0

       # if processed_count_per_id[cam_id][img_id] >= max_images_to_process:
           # continue

        #print(f"Processing {img_name}...")



        # Initialize Mediapipe Face Mesh
        #mp_face_mesh = mp.solutions.face_mesh

        #sufficient_landmarks_found = False

        max_cos_value = -10
        face_most_similar = -111

        for ii in range(tensor_open.shape[0]):

            #model_face_detector.setInputSize([img.shape[1], img.shape[0]])
            #face = model_face_detector.infer(img)

            if max_cos_value < cos_values[ii]:

                max_cos_value = cos_values[ii]
                face_most_similar = ii

                #sufficient_landmarks_found = True
                #score, match = model_face_recognizer.match(target_img, target_faces[0][:-1], img, None)

                cam_id = info_open[ii].split(" ")[0]  # Camera ID
                img_id = info_open[ii].split(" ")[1]

                cam_output_subdir = os.path.join(output_dir, cam_id)
                img_output_subdir = os.path.join(cam_output_subdir, img_id)

                if not os.path.exists(img_output_subdir):
                    os.makedirs(img_output_subdir)


        img = tensor_open[face_most_similar]
        img = img.numpy()
        img_now = np.zeros((shape_face[0], shape_face[1], 3))

        for channel in range(3):
            img_now[:,:,channel] = img[channel, :, :]


        score = cos_values[face_most_similar]

        cv2.imwrite(os.path.join(output_dir, str(i) + ".png"), np.rint(img_now*255))
        file_open.write(info_open[face_most_similar] + " " + str(score) + "\n")

    return True
        #img = visualize_Recognition(img, [0, 0, shape_face - 1, shape_face - 1], target_img, [match], [score])

                #img_name_b = "_".join(info_open[ii].split(" ")[:3])

                #output_img_path = os.path.join(img_output_subdir, f"output_{img_name_b}_{int(score * 100)}.jpg")
                #cv2.imwrite(output_img_path, img)

               # if (cam_id, img_id) not in score_per_id or score > score_per_id[(cam_id, img_id)]:
                   # score_per_id[(cam_id, img_id)] = score

                #if match:
                #    folders_to_rename[img_output_subdir] = os.path.join(cam_output_subdir, target_img_name)

                #break

        #if not sufficient_landmarks_found:
        #    print(f"Skipping {img_name} due to insufficient landmarks.")

    #for original_folder, new_folder in folders_to_rename.items():
    #    if not os.path.exists(new_folder):  # Ensure target folder doesn't already exist
    #        shutil.move(original_folder, new_folder)
    #    else:
    #        print(f"Warning: Target folder '{new_folder}' already exists. Skipping rename for '{original_folder}'.")






def detect_Face_Landmarks(img, mp_face_mesh):
    """
    Detect facial landmarks using Mediapipe Face Mesh and ensure sufficient landmarks are detected.
    Returns True if sufficient landmarks are found, otherwise False.
    """
    cropped_face = (img)
    return True

    if cropped_face.size == 0:
        print("Warning: Cropped face is empty, skipping this face.")
        return False

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        result = face_mesh.process(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
        if result.multi_face_landmarks:
            print("aaaa")
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

        # cv2.putText(frame, score_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
        #             font_thickness)

    return frame