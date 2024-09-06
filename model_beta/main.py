from sface import *
from yunet import *
from detect import *
from input_image_process import *

if __name__ == '__main__':

	import_image()

	input_path = "model_beta/input_folder"
	output_path = "model_beta/output_folder"

	recognizer = SFace(modelPath='model_beta/reg.onnx',
                            disType=0,
                            backendId=cv.dnn.DNN_BACKEND_OPENCV,
                            targetId=cv.dnn.DNN_TARGET_CPU)

	detector = YuNet(modelPath='yunet.onnx',
                          inputSize=[320, 320],
                          confThreshold=0.8,
                          nmsThreshold=0.3,
                          topK=5000,
                          backendId=cv.dnn.DNN_BACKEND_OPENCV,
                          targetId=cv.dnn.DNN_TARGET_CPU)

	detector_tracker = detect_model("model_beta/yolov8n-face.pt")

	cap = cv2.VideoCapture(os.path.join(input_path, video_path))

	count_video_frame = 0

	while cap.isOpened():

		ret, frame = cap.read()

		if not ret:
			break

		frame = detect(detect_model, frame, output_path, count_video_frame)

		frame = process_images(output_path, input_path, detector, recognizer, output_path, frame)

		cv2.imshow("Model", frame)

	cap.release()
	cv2.destroyAllWindows()

















