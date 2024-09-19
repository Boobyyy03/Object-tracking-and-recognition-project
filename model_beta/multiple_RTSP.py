import cv2
from datetime import datetime

# Initialize the camera (0 for the default webcam, or use an IP/RTSP stream URL)
camera_url = 0  # Change this to your camera URL or keep '0' for the default webcam
cap = cv2.VideoCapture(camera_url)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Unable to open the camera")
    exit()

# Video settings
frame_size = (640, 480)  # Resolution of the video
fps = 30  # Frames per second

# Codec and video writer for MP4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_file = 'output_video_with_realtime_timestamp.mp4'
out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

# Start capturing video
print("Press 'q' to stop the recording.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame")
        break

    # Resize frame if needed (optional)
    frame_resized = cv2.resize(frame, frame_size)

    # Get the real-time current date and time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Format as 'YYYY-MM-DD HH:MM:SS.mmm'

    # Add the real-time timestamp to the frame
    cv2.putText(frame_resized, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame with the real-time timestamp
    cv2.imshow('Camera Stream with Real-Time Timestamp', frame_resized)

    # Write the frame to the output file
    out.write(frame_resized)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved to {output_file}")
