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

    for i in range(number_camera):

        tensor_open = pt.load(os.path.join(input_dir, str(i) + ".pt"))
        tensor_open_now = tensor_open.to(device="cpu")
        with open(os.path.join(input_dir, str(i) + ".txt")) as i_o:
            info_open = [line.rstrip() for line in i_o]


        feature_faces = resnet(tensor_open_now)

        feature_target = transform(cv2.resize(target_img, (shape_face[1], shape_face[0])))
        feature_target = feature_target[None, :, :, :]
        feature_target = feature_target.to(device="cpu")
        feature_target = resnet(feature_target)

        cos_values = cos_similarity(feature_faces, feature_target)
        cos_values = cos_values.to(device="cpu")

        max_cos_value = -10
        face_most_similar = -111

        for ii in range(tensor_open.shape[0]):

            if max_cos_value < cos_values[ii]:

                max_cos_value = cos_values[ii]
                face_most_similar = ii

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


        score = cos_values[face_most_similar].detach().item()

        cv2.imwrite(os.path.join(output_dir, str(i) + ".png"), np.rint(img_now*255))
        file_open.write(info_open[face_most_similar] + " " + str(score) + "\n")

    return True

