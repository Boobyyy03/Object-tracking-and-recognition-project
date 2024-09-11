import os
import cv2


def make_Mp4(video_name, fps, detected_frame_dir, project_dir):

    # Lấy hết đường dẫn ảnh trong file
    list_img = list()
    for i in os.listdir(detected_frame_dir):
        list_img.append(i)
    list_img.sort()

    # Lấy các thông số và tạo chỗ đạt file
    cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    img = cv2.imread(detected_frame_dir + os.listdir(detected_frame_dir)[0])
    size = list(img.shape)
    del size[2]
    size.reverse()

    video = cv2.VideoWriter(os.path.join(project_dir, video_name + ".mp4"), cv2_fourcc, fps, size)

    # Viết vào mp4
    for i in list_img:
        video.write(cv2.imread(detected_frame_dir + i))
    video.release()