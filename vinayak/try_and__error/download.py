import requests
import os
def download_file(url, filename):
    if not os.path.exists(filename):
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)

# Download YOLO files
download_file("https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights", "yolov4-tiny.weights")
download_file("https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg", "yolov4-tiny.cfg")
download_file("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names", "coco.names")