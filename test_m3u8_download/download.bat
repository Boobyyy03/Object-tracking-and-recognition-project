ffmpeg -i rtsp://internsys:Them1kynuanhe@nongdanonlnine.ddns.net:554/cam/realmonitor?channel=2^&subtype=0  -c copy -map 0 -f segment -strftime 1 -segment_time 60 -segment_format mp4 "%Y%m%d-%H%M.mp4"
