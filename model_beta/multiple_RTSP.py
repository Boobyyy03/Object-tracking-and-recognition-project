from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

def find_Camera(id):
    cameras = ['rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp',
    'rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp']
    return cameras[int(id)]
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
#  for webcam use zero(0)
 

def gen_Frames(camera_id):
     
    cam = find_Camera(camera_id)
    cap=  cv2.VideoCapture(cam)
    
    while True:
        # for cap in caps:
        # # Capture frame-by-frame
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed/<string:id>/', methods=["GET"])
def video_Feed(id):
   
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()


# sử dụng threading
'''
import threading
import cv2
 
def PlayCamera(id):    
    video_capture = cv2.VideoCapture(id)
    while True:
        
        ret, frame = video_capture.read()
        cv2.imshow('{}'.format(id), frame)        
        
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    video_capture.release()
 
 
 
url1 = "rtsp://admin:"
url2 = "rtsp://admin"
cameraIDs = [0, url1, url2]
 
for id in cameraIDs:
    t = threading.Thread(target=PlayCamera, args=(id,))
    t.start()

    '''