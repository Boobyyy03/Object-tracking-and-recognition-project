import sys
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
import cv2
import os

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
        self.input_dir = r'model_beta\\input_folder'
        self.input_image_dir = r"model_beta\\input_image"
        self.output_dir = r'model_beta\\output_folder'
        self.detected_frame_dir = r'model_beta\\detected_frame_folder'
        self.detect_model_path = r"model_train\\yolov8n-face.pt"

        # Khởi tạo các mô hình nhận diện và phát hiện
        self.face_detector = YuNet(modelPath=r'model_train\\yunet.onnx',
                                   inputSize=[320, 320],
                                   confThreshold=0.8,
                                   nmsThreshold=0.3,
                                   topK=5000,
                                   backendId=cv2.dnn.DNN_BACKEND_OPENCV,
                                   targetId=cv2.dnn.DNN_TARGET_CPU)

        self.face_recognizer = SFace(modelPath=r'model_train\\reg.onnx',
                                     disType=0,
                                     backendId=cv2.dnn.DNN_BACKEND_OPENCV,
                                     targetId=cv2.dnn.DNN_TARGET_CPU)

        self.detect_model_instance = detect_Model(self.detect_model_path, device="cpu")

        # Đảm bảo các thư mục đầu vào và đầu ra tồn tại
        if not os.path.exists(self.input_dir):
            os.makedirs(self.input_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Tạo layout chính
        main_layout = QHBoxLayout()

        # Layout bên trái - Hiển thị video
        self.video_label = QLabel("Video Stream")
        self.video_label.setFixedSize(640, 480)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.video_label)

        # Layout bên phải - Nhập ảnh và hiển thị 3 ảnh xuất ra
        right_layout = QVBoxLayout()

        # Button để nhập ảnh
        self.btn_upload = QPushButton("Chọn ảnh để xử lý")
        self.btn_upload.clicked.connect(self.upload_image)

        # Labels để hiển thị 3 ảnh xuất ra
        self.result_label_1 = QLabel("Ảnh kết quả 1")
        self.result_label_2 = QLabel("Ảnh kết quả 2")
        self.result_label_3 = QLabel("Ảnh kết quả 3")

        right_layout.addWidget(self.btn_upload)
        right_layout.addWidget(self.result_label_1)
        right_layout.addWidget(self.result_label_2)
        right_layout.addWidget(self.result_label_3)

        # Thêm layout trái và phải vào layout chính
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # Đặt layout chính cho cửa sổ
        self.setLayout(main_layout)

        # Khởi động camera để stream video
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Cập nhật mỗi 30ms

        self.target_img_path = None
        self.count_video_frame = 0

    def upload_image(self):
        # Chọn ảnh từ máy tính
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Image Files (*.png *.jpg *.jpeg *.bmp)", options=options)

        if file_name:
            self.target_img_path = file_name
            pixmap = QPixmap(file_name)
            self.result_label_1.setPixmap(pixmap.scaled(200, 200))

            # Sau khi chọn ảnh, thực hiện nhận diện khuôn mặt trên ảnh đó
            process_Images(self.input_dir, self.target_img_path, self.face_detector, self.face_recognizer, self.output_dir)

            # Hiển thị kết quả nhận diện khuôn mặt (nếu có)
            result_image_1_path = os.path.join(self.output_dir, "result_image_1.jpg")
            result_image_2_path = os.path.join(self.output_dir, "result_image_2.jpg")
            result_image_3_path = os.path.join(self.output_dir, "result_image_3.jpg")

            if os.path.exists(result_image_1_path):
                self.result_label_1.setPixmap(QPixmap(result_image_1_path).scaled(200, 200))
            if os.path.exists(result_image_2_path):
                self.result_label_2.setPixmap(QPixmap(result_image_2_path).scaled(200, 200))
            if os.path.exists(result_image_3_path):
                self.result_label_3.setPixmap(QPixmap(result_image_3_path).scaled(200, 200))

    def update_frame(self):
        # Cập nhật frame từ camera
        ret, frame = self.cap.read()
        if ret:
            # Phát hiện khuôn mặt trên video
            detected_frame = detect_Frame(self.detect_model_instance, frame, self.input_dir, self.detected_frame_dir, self.count_video_frame)
            self.count_video_frame += 1

            # Chuyển đổi frame OpenCV sang định dạng QImage
            frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame_rgb.shape
            step = channel * width
            q_img = QImage(frame_rgb.data, width, height, step, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        # Giải phóng camera khi đóng cửa sổ
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("Giao diện xử lý ảnh và video")
    window.show()
    sys.exit(app.exec_())
