import time
import os
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import json

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

def run_ffmpeg(output_dir):
    # Define your ffmpeg command with full paths
    ffmpeg_command = [
        "ffmpeg",
        "-i", "rtsp://internsys:Them1kynuanhe@nongdanonlnine.ddns.net:554/cam/realmonitor?channel=2^&subtype=0",
        "-c", "copy",
        "-map", "0",
        "-f", "segment",
        "-segment_time", "10",
        "-segment_format", "mp4",
        "-segment_list", os.path.join(output_dir, "playlist.m3u8"),
        "-segment_list_flags", "+live",
        "-segment_list_size", "10",
        "-segment_list_type", "m3u8",
        os.path.join(output_dir, "output%03d.mp4")
    ]

    # Run the ffmpeg command
    subprocess.Popen(ffmpeg_command)

if __name__ == "__main__":
    # Read configuration from JSON file
    config_path = "test_m3u8_download/config.json"
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    path = config["path"]  # Directory to monitor
    max_age = config["max_age"]  # Maximum age of files to keep in seconds
    
    # Start ffmpeg in a separate process
    run_ffmpeg(path)

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
