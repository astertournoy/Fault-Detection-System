import time
import os
from datetime import datetime
import cv2
import re
from picamera2 import Picamera2

# -------- INITIAL SETUP --------
# Initialize the Raspberry Pi cameras
picam1 = Picamera2(0)
picam2 = Picamera2(1)

# Directories for saving images
image_directory_left = '/media/atournoy/Expansion/test/left'
image_directory_right = '/media/atournoy/Expansion/test/right'
image_directory_nozzle = '/media/atournoy/Expansion/test/nozzle'

# Create directories if they do not exist
# os.makedirs(image_directory_left, exist_ok=True)
# os.makedirs(image_directory_right, exist_ok=True)
os.makedirs(image_directory_nozzle, exist_ok=True)

# os.environ["QT_QPA_PLATFORM"] = "offscreen"

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

i = 0  # Image counter

# Initialize USB camera (nozzle camera)
nozzle_camera = cv2.VideoCapture("/dev/video16")

try:
    while True:
        i += 1
        current_time = datetime.now()
        timestamp = current_time.strftime("%Y-%m-%dT%H:%M:%S") + f"-{current_time.microsecond // 1000},{current_time.microsecond % 1000}"

        # Generate sanitized file names
        photo_name = f"{i}_{timestamp}.jpg"
        photo_name_left = sanitize_filename(photo_name)
        photo_name_right = sanitize_filename(photo_name)
        photo_name_nozzle = sanitize_filename(photo_name)

        photo_path_left = os.path.join(image_directory_left, photo_name_left)
        photo_path_right = os.path.join(image_directory_right, photo_name_right)
        photo_path_nozzle = os.path.join(image_directory_nozzle, photo_name_nozzle)

        # Capture images from Raspberry Pi cameras
        picam1.start_and_capture_file(photo_path_left)
        picam2.start_and_capture_file(photo_path_right)

        # Capture image from the USB nozzle camera
        ret_nozzle, frame_nozzle = nozzle_camera.read()
        if ret_nozzle:
            cv2.imwrite(photo_path_nozzle, frame_nozzle)
            print("working")

        # Optional: Add a short delay to prevent overloading the system
        time.sleep(0.5)

except KeyboardInterrupt:
    print("Program terminated by user.")

finally:
    # Release the USB camera
    nozzle_camera.release()