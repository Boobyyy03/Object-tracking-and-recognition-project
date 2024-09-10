import sys
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QGridLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
import cv2

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        
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

    def upload_image(self):
        # Chọn ảnh từ máy tính
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Image Files (*.png *.jpg *.jpeg *.bmp)", options=options)
        
        if file_name:
            # Load ảnh và hiển thị trên label (ví dụ ảnh kết quả thứ nhất)
            pixmap = QPixmap(file_name)
            self.result_label_1.setPixmap(pixmap.scaled(200, 200))

    def update_frame(self):
        # Cập nhật frame từ camera
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            step = channel * width
            q_img = QImage(frame.data, width, height, step, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        # Giải phóng camera khi đóng cửa sổ
        self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("Giao diện xử lý ảnh và video")
    window.show()
    sys.exit(app.exec_())
