import sys
import re
from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtGui import QDesktopServices
# Các import liên quan đến phát hiện và nhận diện khuôn mặt
from yunet import YuNet
from sface import SFace
from detect import *
from input_image_process import *
from recognition import *
import shutil
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, \
    QGridLayout, QComboBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
import cv2
import os
import numpy as np

# Các import liên quan đến phát hiện và nhận diện khuôn mặt
from yunet import YuNet
from sface import SFace
from detect import *
from input_image_process import *
from recognition import *


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Định nghĩa các thư mục
        self.input_dir = r'model_beta/input_folder'
        self.input_image_dir = r"model_beta/input_image"
        self.output_dir = r'model_beta/output_folder'
        self.detected_frame_dir = r'model_beta/detected_frame_folder'
        self.detect_model_path = r"model_train/yolov8n-face.pt"
        self.number_camera = 2

        self.current_camera = 0

        # Khởi tạo các mô hình nhận diện và phát hiện
        self.face_detector = YuNet(modelPath=r'model_train/yunet.onnx',
                                   inputSize=[320, 320],
                                   confThreshold=0.8,
                                   nmsThreshold=0.3,
                                   topK=5000,
                                   backendId=cv2.dnn.DNN_BACKEND_OPENCV,
                                   targetId=cv2.dnn.DNN_TARGET_CPU)

        self.face_recognizer = SFace(modelPath=r'model_train/reg.onnx',
                                     disType=0,
                                     backendId=cv2.dnn.DNN_BACKEND_OPENCV,
                                     targetId=cv2.dnn.DNN_TARGET_CPU)

        self.detect_model_instance = list()
        for i in range(self.number_camera):
            self.detect_model_instance.append(detect_Model(self.detect_model_path, device="cpu"))

        # Đảm bảo các thư mục đầu vào và đầu ra tồn tại
        if not os.path.exists(self.input_dir):
            os.makedirs(self.input_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Tạo layout chính
        main_layout = QHBoxLayout()

        # Tạo layout lưới cho 4 video và thêm vào bên trái của giao diện
        left_layout = QVBoxLayout()
        grid_layout = QGridLayout()

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
        self.result_label_1 = QLabel("Ảnh kết quả 1")
        self.result_label_1.setStyleSheet("border:2px solid black;")
        self.result_label_1.setFixedSize(200, 200)

        self.result_label_2 = QLabel("Ảnh kết quả 2")
        self.result_label_2.setStyleSheet("border:2px solid black;")
        self.result_label_2.setFixedSize(200, 200)

        self.result_label_3 = QLabel("Ảnh kết quả 3")
        self.result_label_3.setStyleSheet("border:2px solid black;")
        self.result_label_3.setFixedSize(200, 200)



        right_layout.addWidget(self.btn_upload)
        right_layout.addWidget(self.upload_label)
        right_layout.addWidget(self.result_label_1)
        right_layout.addWidget(self.result_label_2)
        right_layout.addWidget(self.result_label_3)

        right_layout.addWidget(self.btn_mp4)

        self.result_label_1.mousePressEvent = self.on_result_label_1_clicked
        self.result_label_2.mousePressEvent = self.on_result_label_2_clicked

        # Thêm layout bên trái và bên phải vào layout chính
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # Đặt layout chính cho cửa sổ
        self.setLayout(main_layout)

        # Khởi động các video
        self.video_caps = [
            cv2.VideoCapture('video_test/vi2.mp4'),
            cv2.VideoCapture('video_test/vi3.mp4')
        ]

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(30)

        self.camera_box.activated.connect(self.change_camera)

        self.target_img_path = None
        self.count_video_frame = [0] * 2

    def change_camera(self, index):
        self.current_camera = int(index)
        print(index)

    def update_frames(self):
        ret_all = set()

        # Loop through each video label and corresponding video capture
        for i in range(2):
            ret, frame = self.video_caps[i].read()

            if ret:
                # Phát hiện khuôn mặt trên video
                detected_frame = detect_Frame(self.detect_model_instance[i], frame, self.input_dir,
                                              self.detected_frame_dir, i, self.count_video_frame[i])

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

            # Thực hiện recognize
            process_Images(self.input_dir, self.target_img_path, self.face_detector, self.face_recognizer,
                           self.output_dir)

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
                        self.display_scaled_image(self.result_label_1, result_image_1)

                    if cam_id == 1 and len(result_images) > 0:
                        result_image_2 = cv2.imread(result_images[0])  # Assuming the first image in cam_1 folder
                        self.display_scaled_image(self.result_label_2, result_image_2)

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

    def on_result_label_1_clicked(self, event):
        """Handle click on result_label_1 to switch to corresponding camera."""
        if self.result_label_1.pixmap() is not None:
            camera_id = 0  # Camera 1 corresponds to camera_id 0
            self.change_to_camera(camera_id)
        else:
            print("No image in result_label_1")

    def on_result_label_2_clicked(self, event):
        """Handle click on result_label_2 to switch to corresponding camera."""
        if self.result_label_2.pixmap() is not None:
            camera_id = 1  # Camera 2 corresponds to camera_id 1
            self.change_to_camera(camera_id)
        else:
            print("No image in result_label_2")



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
            print("chua co anh")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("Giao diện xử lý ảnh và video")
    window.show()
    sys.exit(app.exec_())
