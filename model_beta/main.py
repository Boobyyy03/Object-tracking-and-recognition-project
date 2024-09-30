import sys
import re
from functools import partial
from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtGui import QDesktopServices
import subprocess
# import imageio_ffmpeg as ffmpeg
import threading
import time
# Các import liên quan đến phát hiện và nhận diện khuôn mặt
from yunet import YuNet
from sface import SFace
from detect import *
from input_image_process import *
from recognition import *
import shutil
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QGridLayout, QComboBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
import cv2
import os
import numpy as np
import torch
# Các import liên quan đến phát hiện và nhận diện khuôn mặt
from yunet import YuNet
from sface import SFace
from detect import *
from input_image_process import *
from recognition import *
from datetime import datetime
from facenet_pytorch import InceptionResnetV1


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Định nghĩa các thư mục
        self.input_dir = r'model_beta/model_beta/input_folder'
        self.input_image_dir = r"model_beta/model_beta/input_image"
        self.output_dir = r'model_beta/model_beta/output_folder'
        self.detected_frame_dir = r'model_beta/model_beta/detected_frame_folder'
        self.detect_model_path = r"model_beta/model_train/yolov8n-face.pt"
        self.target_img_path = None
        self.number_camera = 2

        self.current_camera = 0

        # Khởi tạo các mô hình nhận diện và phát hiện
        self.face_detector = YuNet(modelPath=r'model_beta/model_train/yunet.onnx',
                                   inputSize=[320, 320],
                                   confThreshold=0.8,
                                   nmsThreshold=0.3,
                                   topK=5000,
                                   backendId=cv2.dnn.DNN_BACKEND_OPENCV,
                                   targetId=cv2.dnn.DNN_TARGET_CPU)

        self.face_recognizer = SFace(modelPath=r'model_beta/model_train/reg.onnx',
                                     disType=0,
                                     backendId=cv2.dnn.DNN_BACKEND_OPENCV,
                                     targetId=cv2.dnn.DNN_TARGET_CPU)


        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()


        self.count_dict = list()
        for i in range(self.number_camera):
            self.count_dict.append(0)


        self.detect_model_instance = list()
        for i in range(self.number_camera):
            self.detect_model_instance.append(detect_Model(self.detect_model_path, device="cpu"))

        self.dict_id_images = list()
        for i in range(self.number_camera):
            self.dict_id_images.append(dict())

        for i in range(self.number_camera):
            fileo = open(os.path.join(self.input_dir, str(i) + ".txt"), "w")
            fileo.close()


        # Đảm bảo các thư mục đầu vào và đầu ra tồn tại
        if not os.path.exists(self.input_dir):
            os.makedirs(self.input_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Tạo layout chính
        main_layout = QHBoxLayout()

        # Tạo layout lưới cho 4 video và thêm vào bên trái của giao diện
        left_layout = QVBoxLayout()

        self.video_label = QLabel(f"Video")
        self.video_label.setFixedSize(1280, 960)  # Kích thước nhỏ hơn cho 4 video
        self.video_label.setStyleSheet("border:2px solid black;")

        self.camera_box = QComboBox()
        self.camera_box.addItems([str(i) for i in range(self.number_camera)])

        # Đặt layout lưới vào layout bên trái
        left_layout.addWidget(self.camera_box)
        left_layout.addWidget(self.video_label)

        # Tạo layout bên phải - Nhập ảnh và hiển thị 3 ảnh xuất ra
        right_layout = QVBoxLayout()

        self.upload_label = QLabel("Ảnh mục tiêu")
        self.upload_label.setStyleSheet("border:2px solid black")
        self.upload_label.setFixedSize(200, 200)

        # Button để nhập ảnh
        self.btn_upload = QPushButton("Chọn ảnh để xử lý")
        self.btn_upload.clicked.connect(self.upload_image)

        # Button xuất file mp4
        self.btn_mp4 = QPushButton("Xuất file mp4")
        self.btn_mp4.clicked.connect(self.make_Mp4)

        # Labels để hiển thị 3 ảnh xuất ra

        self.result_labels = list()
        for i in range(self.number_camera):
            result_label = QLabel("Ảnh kết quả " + str(i + 1))
            result_label.setStyleSheet("border:2px solid black;")
            result_label.setFixedSize(200, 200)
            self.result_labels.append(result_label)



        right_layout.addWidget(self.btn_upload)
        right_layout.addWidget(self.upload_label)
        for i in range(self.number_camera):
            right_layout.addWidget(self.result_labels[i])

        right_layout.addWidget(self.btn_mp4)


        for i in range(self.number_camera):
            self.result_labels[i].mousePressEvent = partial(self.on_result_label_clicked, i)


        box_layout = QVBoxLayout()

        # box để cân chỉnh giao diện cho dễ nhìn
        box_blank = QLabel("")
        box_blank.setFixedSize(200, 280)
        box_layout.addWidget(box_blank)

        self.box_results = list()

        # Thêm một biến điều kiện để đảm bảo an toàn cho luồng
        self.stop_thread = False
        # Tạo và khởi chạy luồng khi khởi tạo đối tượng MainWindow
        self.thread = threading.Thread(target=self.create_segments_m3u8)
        self.thread.start()

        for i in range(self.number_camera):
            box_result = QLabel("")
            box_result.setFixedSize(200, 200)
            box_result.setStyleSheet("border:2px solid black")
            self.box_results.append(box_result)
            box_layout.addWidget(self.box_results[i])


        box_blank2 = QLabel("")
        box_blank2.setFixedSize(200, 20)
        box_layout.addWidget(box_blank2)

        # Thêm layout bên trái và bên phải vào layout chính
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        main_layout.addLayout(box_layout)

        # Đặt layout chính cho cửa sổ
        self.setLayout(main_layout)

        # Khởi động các video
        self.video_caps = [
            cv2.VideoCapture('model_beta/video_test/video.mp4'),
            cv2.VideoCapture('model_beta/video_test/vi2.mp4')
        ]

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(1500)

        self.camera_box.activated.connect(self.change_camera)

        self.target_img_path = None
        self.count_video_frame = [0] * 2

    def display_box_text(self,id_cam, score):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # for i in range(self.number_camera):
        # # Set the box text with the camera number and current date/time
        self.box_results[id_cam].setText(f"Camera {str(id_cam + 1)}\nTime: {current_time}\nConfidence: {score}")

    def change_camera(self, index):
        self.current_camera = int(index)
        print(self.current_camera)

        if self.target_img_path:

            image = cv2.imread(self.target_img_path)

            self.recognite_image(image)
        

    def update_frames(self):
        ret_all = set()
        
        # Loop through each video label and corresponding video capture
        for i in range(self.number_camera):
            ret, frame = self.video_caps[i].read()

            if ret:
                # Phát hiện khuôn mặt trên video
                detected_frame, self.dict_id_images[i], self.count_dict[i] = detect_Frame(self.detect_model_instance[i], frame, self.dict_id_images[i], self.count_dict[i], self.resnet, self.input_dir,
                                              self.detected_frame_dir, i, self.count_video_frame[i])

                # Tạo thư mục phụ theo id camera nếu chưa tồn tại
                camera_dir = os.path.join(self.detected_frame_dir, str(i))
                if not os.path.exists(camera_dir):
                    os.makedirs(camera_dir)

                # # Lưu frame đã phát hiện khuôn mặt vào thư mục phụ
                # cv2.imwrite(os.path.join(camera_dir, f"frame_{self.count_video_frame[i]}.jpg"), frame)

                self.count_video_frame[i] += 1

                # Cập nhật video hiển thị
                if self.current_camera == i:
                    self.display_frame(self.video_label, detected_frame)
            else:
                # If the video ends, stop updating that video
                ret_all.add(i)

        # If all videos have ended, stop the timer
        if len(ret_all) == len(self.video_caps):
            self.timer.stop()

    def create_segment(self, cam_id, frame_files):
        """Tạo một đoạn video segment từ danh sách frame sử dụng FFmpeg trực tiếp."""
        fps = 24  # Giả sử là 24 fps
        try:
            # Đường dẫn để lưu segment
            segment_dir = os.path.join("model_beta", "segments", str(cam_id))
            os.makedirs(segment_dir, exist_ok=True)
            segment_file = os.path.join(segment_dir, f"segment_{int(time.time())}.mp4")

            # Đường dẫn đến thư mục chứa các frame
            detected_frame_dir = os.path.join(self.detected_frame_dir, str(cam_id))

            # Lọc các file có đuôi .png và bắt đầu bằng 6 chữ số
            frame_files = sorted([f for f in os.listdir(detected_frame_dir) 
                                if f.endswith('.png') and f[:6].isdigit()])

            # Kiểm tra xem có frame nào không
            if not frame_files:
                print("Không có frame nào để tạo segment.")
                return

            # Tạo danh sách các đường dẫn đến file frame
            frame_paths = [os.path.join(detected_frame_dir, f) for f in frame_files]

            # Thiết lập lệnh FFmpeg để tạo video từ file PNG
            ffmpeg_cmd = [
                'ffmpeg',  # Gọi FFmpeg trực tiếp
                '-y',  # Ghi đè file nếu đã tồn tại
                '-framerate', str(fps),  # Đặt FPS
                '-i', 'concat:' + '|'.join(frame_paths),  # Đầu vào các frame
                '-c:v', 'libx264',  # Codec video
                '-pix_fmt', 'yuv420p',  # Định dạng màu
                segment_file  # Đầu ra video
            ]

            # Chạy lệnh FFmpeg để tạo video
            process = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Kiểm tra nếu FFmpeg gặp lỗi
            if process.returncode != 0:
                print(f"Lỗi FFmpeg: {process.stderr.decode('utf-8')}")
                return

            print(f"Segment created: {segment_file}")

            # Cập nhật file playlist M3U8
            self.update_m3u8_playlist(cam_id, segment_file)

        except Exception as e:
            print(f"Lỗi khi tạo segment: {e}")


    def create_segments_m3u8(self):
        """Tạo các đoạn segment và playlist M3U8 khi đủ số lượng frame."""
        while not self.stop_thread:
            try:
                # Kiểm tra số lượng frame trong thư mục detected_frame_folder
                for cam_id in range(self.number_camera):
                    detected_frame_dir = os.path.join(self.detected_frame_dir, str(cam_id))

                    if not os.path.exists(detected_frame_dir):
                        print(f"Thư mục {detected_frame_dir} không tồn tại.")
                        continue

                    # Lấy tất cả các file trong thư mục
                    all_files = os.listdir(detected_frame_dir)

                    # Lọc các file .png
                    frame_files = sorted([f for f in all_files if f.endswith('.png')])

                    # Nếu không có đủ số lượng frame để tạo segment
                    if len(frame_files) < 10:
                        print(f"Không đủ số lượng frame để tạo segment.")
                        continue

                    # Tạo segment từ các frame
                    self.create_segment(cam_id, frame_files)

                    # Sau khi tạo segment, xóa các frame đã dùng
                    for frame_file in frame_files[:10]:
                        try:
                            os.remove(os.path.join(detected_frame_dir, frame_file))
                        except FileNotFoundError:
                            print(f"File không tồn tại hoặc đã bị xóa: {frame_file}")

                # Đợi một khoảng thời gian ngắn trước khi kiểm tra lại
                time.sleep(5)  # Đợi 5 giây trước khi kiểm tra lại
            except Exception as e:
                print(f"Lỗi khi tạo segment: {e}")

    def update_m3u8_playlist(self, cam_id, segment_file):
        """Cập nhật file playlist M3U8."""
        playlist_path = os.path.join("model_beta", "segments", f"playlist_{cam_id}.m3u8")
        with open(playlist_path, 'a') as playlist:
            playlist.write(f"#EXTINF:10.0,\n{os.path.basename(segment_file)}\n")

    def display_frame(self, label, frame):
        """Display a frame in a QLabel."""
        height, width, channel = frame.shape
        layout_width, layout_height = label.width(), label.height()

        # Tính tỷ lệ khung hình của frame và layout
        frame_aspect_ratio = width / height
        layout_aspect_ratio = layout_width / layout_height

        if frame_aspect_ratio > layout_aspect_ratio:
            # Video rộng hơn so với layout
            new_width = layout_width
            new_height = int(new_width / frame_aspect_ratio)
        else:
            # Video cao hơn hoặc cùng tỷ lệ với layout
            new_height = layout_height
            new_width = int(new_height * frame_aspect_ratio)

        # Resize frame to new dimensions
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Create a black image of layout size
        black_frame = np.zeros((layout_height, layout_width, 3), dtype=np.uint8)

        # Calculate the position to place the resized video in the center of the black frame
        x_offset = (layout_width - new_width) // 2
        y_offset = (layout_height - new_height) // 2

        # Place the resized video on the black frame
        black_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame

        # Convert the black_frame to QImage
        frame_rgb = cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB)
        step = 3 * layout_width
        q_img = QImage(frame_rgb.data, layout_width, layout_height, step, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q_img))

    def upload_image(self):
        # Chọn ảnh từ máy tính
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Image Files (*.png *.jpg *.jpeg *.bmp)",
                                                   options=options)

        if file_name:
            self.target_img_path = file_name

            image = cv2.imread(file_name)

            self.recognite_image(image)

    def recognite_image(self, image):
        # Thực hiện recognize
        process_Images(self.input_dir, self.target_img_path, self.face_detector, self.face_recognizer, self.output_dir)

        # Scale và hiện ảnh tải lên
        self.display_scaled_image(self.upload_label, image)

        # Lấy tên file của input_image hoặc target_img_path để khớp với folder trong cam folder
        input_image_name = os.path.basename(self.target_img_path).split('.')[
            0]  # Ex: "target_img_path.jpg" -> "target_img_path"
        print(input_image_name)

        # Iterate through each camera folder in output_folder
        for cam_id in range(self.number_camera):
            cam_folder = os.path.join(self.output_dir,
                                      f"{cam_id}")  # Ensure 'cam_' prefix matches your folder structure

            # Check if a folder with the input image name exists inside the camera folder
            img_folder = os.path.join(cam_folder, input_image_name)
            if os.path.exists(img_folder):
                # Load images from the folder
                result_images = sorted([os.path.join(img_folder, img) for img in os.listdir(img_folder)
                                        if img.endswith(('.jpg', '.png', '.jpeg', '.bmp'))])

                # Display results in the respective result labels
                if cam_id == 0 and len(result_images) > 0:
                    result_image_1 = cv2.imread(result_images[0])
                    self.display_scaled_image(self.result_labels[0], result_image_1)
                    confScore = result_images[0].split("_")[-1].split(".")[0]
                    self.display_box_text(cam_id, confScore)
                if cam_id == 1 and len(result_images) > 0:
                    result_image_2 = cv2.imread(result_images[0])  # Assuming the first image in cam_1 folder
                    self.display_scaled_image(self.result_labels[1], result_image_2)
                    confScore = result_images[0].split("_")[-1].split(".")[0]
                    self.display_box_text(cam_id, confScore)

    def display_scaled_image(self, label, image):
        """Scale the image to fit inside the QLabel and pad with black if necessary."""
        height, width, channel = image.shape
        layout_width, layout_height = label.width(), label.height()

        # Tính tỷ lệ khung hình của image và layout
        image_aspect_ratio = width / height
        layout_aspect_ratio = layout_width / layout_height

        if image_aspect_ratio > layout_aspect_ratio:
            # Image rộng hơn so với layout
            new_width = layout_width
            new_height = int(new_width / image_aspect_ratio)
        else:
            # Image cao hơn hoặc cùng tỷ lệ với layout
            new_height = layout_height
            new_width = int(new_height * image_aspect_ratio)

        # Resize image to new dimensions
        resized_image = cv2.resize(image, (new_width, new_height))

        # Create a black image of layout size
        black_frame = np.zeros((layout_height, layout_width, 3), dtype=np.uint8)

        # Calculate the position to place the resized image in the center of the black frame
        x_offset = (layout_width - new_width) // 2
        y_offset = (layout_height - new_height) // 2

        # Place the resized image on the black frame
        black_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

        # Convert the black_frame to QImage
        frame_rgb = cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB)
        step = 3 * layout_width
        q_img = QImage(frame_rgb.data, layout_width, layout_height, step, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        """Dừng luồng khi đóng cửa sổ."""
        self.stop_thread = True
        self.thread.join()  # Đợi luồng dừng trước khi đóng chương trình

        # Giải phóng các video khi đóng cửa sổ
        for cap in self.video_caps:
            cap.release()
        cv2.destroyAllWindows()

    def change_to_camera(self, camera_id):
        """Chuyển đến camera chứa người được nhận diện."""
        self.current_camera = camera_id
        print(f"Chuyển đến Camera {camera_id + 1}")
        # Optional: Add code to update UI or refresh video display when the camera changes

    def extract_camera_id(self, filename):
        """Trích xuất ID camera từ tên tệp."""
        match = re.search(r'output_(\d+)_', filename)
        if match:
            return int(match.group(1))
        return None

    def on_result_label_clicked(self, number, event):
        """Handle click on result_label to switch to corresponding camera."""
        if self.result_labels[number].pixmap() is not None:
            camera_id = number
            self.change_to_camera(camera_id)
        else:
            print(f"No image in result_label_{number + 1}")


    def make_Mp4(self):
        fps = 24
        try:
            # Lấy hết đường dẫn ảnh trong file
            for i in range(self.number_camera):
                detected_frame_dir = os.path.join(self.detected_frame_dir, str(i))
                list_img = list()
                for ii in os.listdir(detected_frame_dir):
                    list_img.append(ii)
                list_img.sort()
                print(i)

                # Lấy các thông số và tạo chỗ đạt file
                cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                img = cv2.imread(os.path.join(detected_frame_dir, os.listdir(detected_frame_dir)[0]))
                print(i)

                size = list(img.shape)
                del size[2]
                size.reverse()
                print(i)

                video = cv2.VideoWriter(os.path.join("model_beta", str(i) + ".mp4"), cv2_fourcc, fps, size)
                print(i)

                # Viết vào mp4
                for ii in list_img:
                    video.write(cv2.imread(os.path.join(detected_frame_dir, ii)))
                video.release()
                print(i)
        except:
            print("Chưa có ảnh để tạo video")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("Giao diện xử lý ảnh và video")
    window.show()
    sys.exit(app.exec_())