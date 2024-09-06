import cv2
from sface import SFace
from yunet import YuNet
from detect import detect_model, detect
from input_image_process import import_image
from recognition import process_images
import os


if __name__ == '__main__':

    # Nhập ảnh vào thư mục input
    import_image()

    # Đặt đường dẫn thư mục
    input_path = "model_beta/input_image"
    output_path = "model_beta/output_folder"
    
    # Khởi tạo bộ nhận diện khuôn mặt
    recognizer = SFace(modelPath=r'C:\Users\Administrator\Desktop\model_beta\reg.onnx',
                       disType=0,
                       backendId=cv2.dnn.DNN_BACKEND_OPENCV,
                       targetId=cv2.dnn.DNN_TARGET_CPU)

    # Khởi tạo bộ phát hiện khuôn mặt bằng YuNet
    detector = YuNet(modelPath='yunet.onnx',
                     inputSize=[320, 320],
                     confThreshold=0.8,
                     nmsThreshold=0.3,
                     topK=5000,
                     backendId=cv2.dnn.DNN_BACKEND_OPENCV,
                     targetId=cv2.dnn.DNN_TARGET_CPU)

    # Khởi tạo mô hình YOLO
    detector_tracker = detect_model(r"C:\Users\Administrator\Desktop\model_beta\yolov8n-face.pt")

    # Đọc video từ thư mục input
    video_path = r"C:\Users\Administrator\Desktop\model_beta\video1.mp4"  # Thay đổi tên file video ở đây
    cap = cv2.VideoCapture(os.path.join(input_path, video_path))

    count_video_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Phát hiện và cắt khuôn mặt từ khung hình
        frame = detect(detector_tracker, frame, output_path, count_video_frame)

        # Xử lý ảnh với YuNet và SFace
        frame = process_images(os.path.join(output_path, str(1) + "_" + str(count_video_frame) + ".png"), input_path, detector, recognizer, output_path)

        # Hiển thị khung hình
        cv2.imshow("Model", frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        count_video_frame += 1

    cap.release()
    cv2.destroyAllWindows()
