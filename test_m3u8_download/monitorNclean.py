import time
import os
import subprocess
import json
import csv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class EventHandler(FileSystemEventHandler):
    def __init__(self, segment_dir, max_age):
        self.segment_dir = segment_dir
        self.max_age = max_age

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.mp4'):
            print(f"Detected new file: {event.src_path}, running cleanup...")
            self.cleanup_old_files()

    def cleanup_old_files(self):
        now = time.time()
        for filename in os.listdir(self.segment_dir):
            file_path = os.path.join(self.segment_dir, filename)
            if filename.endswith('.mp4') and os.path.isfile(file_path):
                file_age = now - os.path.getmtime(file_path)
                if file_age > self.max_age:
                    print(f"Deleting old file: {file_path}")
                    os.remove(file_path)

def run_ffmpeg(output_dir, csv_file):
    # Open the CSV file to append timestamps
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Timestamp'])  # Write the header if needed

        # Define your ffmpeg command with full paths
        ffmpeg_command = [
            "ffmpeg",
            "-i", "rtsp://internsys:Them1kynuanhe@nongdanonlnine.ddns.net:554/cam/realmonitor?channel=2^&subtype=0",
            "-c", "copy",
            "-map", "0",
            "-f", "segment",
            "-segment_time", "10",
            "-segment_format", "mp4",
            "-strftime", "1",  # Enable strftime format for output file names
            "-segment_list", os.path.join(output_dir, "playlist.m3u8"),
            "-segment_list_flags", "+live",
            "-segment_list_size", "10",
            "-segment_list_type", "m3u8",
            os.path.join(output_dir, "output_%Y-%m-%d_%H-%M-%S.mp4")
        ]

        # Start the ffmpeg process and capture stdout
        process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Monitor ffmpeg output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Check for segment creation messages in ffmpeg output
                if "Opening" in output and ".mp4" in output:
                    # Extract the file name and current timestamp
                    filename = output.split("'")[1]
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    print(f"Segment created: {filename}, Timestamp: {timestamp}")
                    
                    # Write to CSV
                    writer.writerow([filename, timestamp])

        process.wait()

if __name__ == "__main__":
    # Read configuration from JSON file
    config_path = "test_m3u8_download/config.json"
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    path = config["path"]  # Directory to monitor
    max_age = config["max_age"]  # Maximum age of files to keep in seconds
    csv_file = os.path.join(path, "segments_timestamps.csv")  # CSV file to store timestamps
    
    # Start ffmpeg in a separate process
    run_ffmpeg(path, csv_file)

    # Set up file monitoring
    event_handler = EventHandler(path, max_age)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
