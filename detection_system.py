import time
import os
from datetime import datetime
import cv2
import csv
import re
from ultralytics import YOLO  # Import YOLO for classification
from picamera2 import Picamera2  # Import Raspberry Pi camera library

# -------- INITIAL SETUP --------
# Initialize the Raspberry Pi cameras
picam1 = Picamera2(0)  # Left camera (Raspberry Pi)
picam2 = Picamera2(1)  # Right camera (Raspberry Pi)
nozzle_camera = cv2.VideoCapture("/dev/video16")  # Nozzle camera (USB)

# Directories for saving images
image_directory_left = '/media/atournoy/Expansion/test/left'
image_directory_right = '/media/atournoy/Expansion/test/right'
image_directory_nozzle = '/media/atournoy/Expansion/test/nozzle'

# Create directories if they do not exist
os.makedirs(image_directory_left, exist_ok=True)
os.makedirs(image_directory_right, exist_ok=True)
os.makedirs(image_directory_nozzle, exist_ok=True)

# Load YOLO models
model_side = YOLO("models/side/weights/best.pt")
model_nozzle = YOLO("models/nozzle/weights/best.pt")

# Lists to store classification results
results_left = []
results_right = []
results_nozzle = []

#counter of good answers
Lg = 0
Rg = 0
Ng = 0

# -------- CROPPING --------
# Global variables for cropping (side cameras)
is_dragging = False
# Add separate crop box variables for left and right cameras
left_box_x, left_box_y, left_crop_w, left_crop_h = 0, 0, 1600, 1000
right_box_x, right_box_y, right_crop_w, right_crop_h = 0, 0, 1600, 1000
display_scale = 1  # Scale factor for displaying the image
crop_set = False  # Flag to track if cropping has been set

# Define the fixed crop values for nozzle camera
crop_x, crop_y, crop_w_nozzle, crop_h_nozzle = 145, 135, 440, 320


def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def mouse_callback_left(event, x, y, flags, param):
    global is_dragging, left_box_x, left_box_y, left_crop_w, left_crop_h
    scaled_x, scaled_y = int(x / display_scale), int(y / display_scale)
    if event == cv2.EVENT_LBUTTONDOWN:
        is_dragging = True
        left_box_x, left_box_y = scaled_x - left_crop_w // 2, scaled_y - left_crop_h // 2
    elif event == cv2.EVENT_MOUSEMOVE and is_dragging:
        left_box_x, left_box_y = scaled_x - left_crop_w // 2, scaled_y - left_crop_h // 2
    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging = False

def mouse_callback_right(event, x, y, flags, param):
    global is_dragging, right_box_x, right_box_y, right_crop_w, right_crop_h
    scaled_x, scaled_y = int(x / display_scale), int(y / display_scale)
    if event == cv2.EVENT_LBUTTONDOWN:
        is_dragging = True
        right_box_x, right_box_y = scaled_x - right_crop_w // 2, scaled_y - right_crop_h // 2
    elif event == cv2.EVENT_MOUSEMOVE and is_dragging:
        right_box_x, right_box_y = scaled_x - right_crop_w // 2, scaled_y - right_crop_h // 2
    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging = False

def crop_side_image(img_path, camera_name, is_left_camera):
    global left_box_x, left_box_y, left_crop_w, left_crop_h
    global right_box_x, right_box_y, right_crop_w, right_crop_h
    global display_scale

    img = cv2.imread(img_path)
    if img is not None:
        display_img = cv2.resize(img, None, fx=display_scale, fy=display_scale)
        cv2.namedWindow(camera_name)
        if is_left_camera:
            cv2.setMouseCallback(camera_name, mouse_callback_left)
        else:
            cv2.setMouseCallback(camera_name, mouse_callback_right)

        while True:
            temp_display_img = display_img.copy()
            if is_left_camera:
                display_box_x = int(left_box_x * display_scale)
                display_box_y = int(left_box_y * display_scale)
                display_crop_w = int(left_crop_w * display_scale)
                display_crop_h = int(left_crop_h * display_scale)
            else:
                display_box_x = int(right_box_x * display_scale)
                display_box_y = int(right_box_y * display_scale)
                display_crop_w = int(right_crop_w * display_scale)
                display_crop_h = int(right_crop_h * display_scale)

            cv2.rectangle(temp_display_img, (display_box_x, display_box_y),
                          (display_box_x + display_crop_w, display_box_y + display_crop_h),
                          (0, 255, 0), 2)
            cv2.imshow(camera_name, temp_display_img)
            key = cv2.waitKey(1)
            if key == ord('c'):  # Confirm cropping
                if is_left_camera:
                    cropped_img = img[left_box_y:left_box_y + left_crop_h, left_box_x:left_box_x + left_crop_w]
                else:
                    cropped_img = img[right_box_y:right_box_y + right_crop_h, right_box_x:right_box_x + right_crop_w]
                cv2.imwrite(img_path, cropped_img)
                break
        cv2.destroyAllWindows()
        # -----

def apply_crop(img_path, is_left_camera):
    global left_box_x, left_box_y, left_crop_w, left_crop_h
    global right_box_x, right_box_y, right_crop_w, right_crop_h

    img = cv2.imread(img_path)
    if img is not None:
        if is_left_camera:
            cropped_img = img[left_box_y:left_box_y + left_crop_h, left_box_x:left_box_x + left_crop_w]
        else:
            cropped_img = img[right_box_y:right_box_y + right_crop_h, right_box_x:right_box_x + right_crop_w]
        cv2.imwrite(img_path, cropped_img)


# -------- LOGIC FRAMEWORK --------
def check_consecutive(lst, error_type, count):
    # Returns True if 'error_type' appears 'count' times consecutively in lst
    return len(lst) >= count and all(item == error_type for item in lst[-count:])


#function of the logic
def answers(results_left, results_right, results_nozzle):
    L_pred = results_left[-1]
    L_pred = L_pred[-1]
    R_pred = results_right[-1]
    R_pred = R_pred[-1]
    N_pred = results_nozzle[-1]
    N_pred = N_pred[-1]
    print("predictions: ", L_pred, R_pred, N_pred)

    L_ans = "none"
    R_ans = "none"
    N_ans = "none"

    global Lg
    global Rg
    global Ng


    # in case the model returns 'good', 'unclear' or 'not printing'
    if L_pred == '01_Good' or L_pred == '14_Unclear' or L_pred == '15_Not_Printing':
        L_ans = L_pred
        if L_pred == '01_Good':
            results_left = results_left[:-1]
            Lg += 1
        if L_pred == '14_Unclear':
            results_left = results_left[:-1]
        if L_pred == '15_Not_Printing':
            results_left = results_left[:-1]
            L_ans = "none"

    if R_pred in ['01_Good', '14_Unclear', '15_Not_Printing']:
        R_ans = R_pred
        if R_pred == '01_Good':
            results_right = results_right[:-1]
            Rg += 1
        if R_pred == '14_Unclear':
            results_right = results_right[:-1]
        if R_pred == '15_Not_Printing':
            results_right = results_right[:-1]
            R_ans = "none"

    if N_pred in ['01_Good', '14_Unclear', '15_Not_Printing']:
        N_ans = N_pred
        if N_pred == '01_Good':
            results_nozzle = results_nozzle[:-1]
            Ng += 1
        if N_pred == '14_Unclear':
            results_nozzle = results_nozzle[:-1]
        if N_pred == '15_Not_Printing':
            results_nozzle = results_nozzle[:-1]
            N_ans = "none"

    #when good appear 4x, the list is emptied
    if Lg >= 4:
        results_left = ['none']
        Lg = 0
    if Rg >= 4:
        results_right = ['none']
        Rg = 0
    if Ng >= 4:
        results_nozzle = ['none']
        Ng = 0



    #when 4 times in a row the same error:
    if check_consecutive(results_left, L_pred, 4):
        L_ans = L_pred
        Lg = 0

    if check_consecutive(results_right, R_pred,4):
        R_ans = R_pred
        Rg = 0

    if check_consecutive(results_right, N_pred,4):
        N_ans = N_pred
        Ng = 0

    
    #when 2 times in a row the same error from 2 cameras simultanuously:
    if check_consecutive(results_left,L_pred,2) and check_consecutive(results_right,L_pred,2):
        L_ans = L_pred
        R_ans = L_pred
        Lg = 0
        Rg = 0
    
    if check_consecutive(results_left,L_pred,2) and check_consecutive(results_nozzle,L_pred,2):
        L_ans = L_pred
        N_ans = L_pred
        Lg = 0
        Ng = 0

    if check_consecutive(results_right,R_pred,2) and check_consecutive(results_nozzle,R_pred,2):
        R_ans = R_pred
        N_ans = R_pred
        Rg = 0
        Ng = 0


    #when all 3 cameras say the same error:
    if L_pred == R_pred and R_pred == N_pred:
        L_ans = N_pred
        R_ans = R_pred
        N_ans = L_pred
        Lg = 0
        Rg = 0
        Ng = 0



    #special cases
    # not printing
    if L_pred == N_pred == '15_Not_Printing':
        L_ans = L_pred
        N_ans = L_pred

    if R_pred == N_pred == '15_Not_Printing':
        R_ans = R_pred
        N_ans = N_pred

    if L_pred == R_pred == "15_Not_Printing":
        L_ans = L_pred
        R_ans = R_pred


    #overhang sag, bridging, delamination and OE
    if (L_ans in ['09_Poor_Bridging', '10_Overhang_Sag', '13_Delamination']) and (N_ans not in ['01_Good', '14_Unclear', '15_Not_Printing']):
        N_ans = 'none'

    if (R_ans in ['09_Poor_Bridging', '10_Overhang_Sag', '13_Delamination']) and (N_ans not in ['01_Good', '14_Unclear', '15_Not_Printing']):
        N_ans = 'none'

    if (N_ans == '02_Over_Extrusion' or N_ans == '03_Under_Extrusion') and (L_ans not in ['01_Good', '14_Unclear', '15_Not_Printing']):
        L_ans = 'none'

    if (N_ans == '02_Over_Extrusion' or N_ans == '03_Under_Extrusion') and (R_ans not in ['01_Good', '14_Unclear', '15_Not_Printing']):
        R_ans = 'none'

    return(L_ans, R_ans, N_ans)

def update_results_csv(T_ans):
    # Path to the CSV file
    csv_file = "results.csv"
    
    # Check if the CSV file already exists. If not, create it and add headers
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # If the file doesn't exist, add a header row
        if not file_exists:
            writer.writerow(["Conclusion"])  # You can add more columns if needed
        
        # Append the current T_ans to the file
        writer.writerow([T_ans])

# integrate three camera outputs into one conclusion
def conclusion (L_ans, R_ans, N_ans):
    print('according to the Nozzle camera: ',N_ans)
    print('according to the Left camera: ', L_ans)
    print('according to the Right camera: ', R_ans)

    #when all three answers are the same
    if N_ans == L_ans and L_ans == R_ans:
        if N_ans in ['none']:
            T_ans = N_ans
            print('CONCLUSION: ', T_ans)
        else:
            T_ans = N_ans
            print('CONCLUSION: ', T_ans)


    #when all of the answers are either good or unclear
    elif N_ans in ['01_Good', '14_Unclear', 'none'] and L_ans in ['01_Good', '14_Unclear', 'none'] and R_ans in ['01_Good', '14_Unclear', 'none']:
        T_ans = '01_Good'
        print('CONCLUSION: ', T_ans)

    #when only one of the cameras shows an error
    elif N_ans not in ['01_Good', '14_Unclear', 'none'] and L_ans in ['01_Good', '14_Unclear', 'none'] and R_ans in ['01_Good', '14_Unclear', 'none']:
        T_ans = N_ans
        print('CONCLUSION: ', T_ans)

    elif N_ans in ['01_Good', '14_Unclear', 'none'] and L_ans not in ['01_Good', '14_Unclear', 'none'] and R_ans in ['01_Good', '14_Unclear', 'none']:
        T_ans = L_ans
        print('CONCLUSION: ', T_ans)

    elif N_ans in ['01_Good', '14_Unclear', 'none'] and L_ans in ['01_Good', '14_Unclear', 'none'] and R_ans not in ['01_Good', '14_Unclear', 'none']:
        T_ans = R_ans
        print('CONCLUSION: ', T_ans)

    #when two cameras show errors
    #nozzle is good others show error
    elif N_ans in ['01_Good', '14_Unclear', 'none'] and L_ans not in ['01_Good', '14_Unclear', 'none'] and R_ans not in ['01_Good', '14_Unclear', 'none']:
        if L_ans == R_ans:
            T_ans = L_ans
            print('CONCLUSION: ', T_ans)

        else:
            T_ans = 'Multiple Errors'
            print('CONCLUSION MULTIPLE ERRORS: ', L_ans, ' AND ', R_ans)
     
    #right camera is good, others show error
    elif N_ans not in ['01_Good', '14_Unclear', 'none'] and L_ans not in ['01_Good', '14_Unclear', 'none'] and R_ans in ['01_Good', '14_Unclear', 'none']:
        if L_ans == N_ans:
            T_ans = L_ans
            print('CONCLUSION: ', T_ans)

        else:
            T_ans = 'Multiple Errors'
            print('CONCLUSION MULTIPLE ERRORS: ', L_ans, ' AND ', N_ans)


    #left camera is good, others show error
    elif N_ans not in ['01_Good', '14_Unclear', 'none'] and L_ans in ['01_Good', '14_Unclear', 'none'] and R_ans not in ['01_Good', '14_Unclear', 'none']:
        if R_ans == N_ans:
            T_ans = R_ans
            print('CONCLUSION: ', T_ans)

        else:
            T_ans = 'Multiple Errors'
            print('CONCLUSION MULTIPLE ERRORS: ', R_ans, ' AND ', N_ans)


    #when all three show erros
    elif N_ans not in ['01_Good', '14_Unclear', 'none'] and L_ans not in ['01_Good', '14_Unclear', 'none'] and R_ans not in ['01_Good', '14_Unclear', 'none']:
        if N_ans == L_ans:
            T_ans = 'Multiple Errors'
            print('CONCLUSION MULTIPLE ERRORS: ', R_ans, ' AND ', N_ans)

        elif N_ans == R_ans:
            T_ans = 'Multiple Errors'
            print('CONCLUSION MULTIPLE ERRORS: ', L_ans, ' AND ', N_ans)

        elif L_ans == R_ans:
            T_ans = 'Multiple Errors'
            print('CONCLUSION MULTIPLE ERRORS: ', L_ans, ' AND ', N_ans)



    update_results_csv(T_ans)

    return T_ans







# -------- MAIN FUNCTIONS --------
left_image = None
right_image = None

def capture_and_classify_images():
    global crop_set, results_left, results_right, results_nozzle
    i = 0  # Image counter
    while True:
        i += 1
        current_time = datetime.now()
        timestamp = current_time.strftime("%Y-%m-%dT%H:%M:%S") + f"-{current_time.microsecond // 1000},{current_time.microsecond % 1000}"

        photo_name = f"{i}_{timestamp}.jpg"
        photo_name_left = sanitize_filename(photo_name)
        photo_name_right = sanitize_filename(photo_name)
        photo_name_nozzle = sanitize_filename(photo_name)

        photo_path_left = os.path.join(image_directory_left, photo_name_left)
        photo_path_right = os.path.join(image_directory_right, photo_name_right)
        photo_path_nozzle = os.path.join(image_directory_nozzle, photo_name_nozzle)

        # Capture images from Raspberry Pi cameras
        picam1.start_and_capture_file(photo_path_left, preview_mode=None, capture_mode='still', show_preview=False)
        left_image = cv2.imread(photo_path_left)
        left_image = cv2.flip(left_image, 0)
        picam2.start_and_capture_file(photo_path_right, preview_mode=None, capture_mode='still', show_preview=False)
        right_image = cv2.imread(photo_path_right)
        right_image = cv2.flip(right_image, 0)
        ret_nozzle, frame_nozzle = nozzle_camera.read()

        if left_image is not None and right_image is not None and ret_nozzle:

            cv2.imwrite(photo_path_left, left_image)
            cv2.imwrite(photo_path_right, right_image)
            cv2.imwrite(photo_path_nozzle, frame_nozzle)

            # Crop nozzle image
            nozzle_image = cv2.imread(photo_path_nozzle)
            cropped_nozzle_img = nozzle_image[crop_y:crop_y + crop_h_nozzle, crop_x:crop_x + crop_w_nozzle]
            cv2.imwrite(photo_path_nozzle, cropped_nozzle_img)

            # Crop side images
            if not crop_set:
                crop_side_image(photo_path_left, "Left Camera", is_left_camera=True)
                crop_side_image(photo_path_right, "Right Camera", is_left_camera=False)
                crop_set = True
            else:
                apply_crop(photo_path_left, is_left_camera=True)
                apply_crop(photo_path_right, is_left_camera=False)

            # Classify images
            classify_image_side(photo_path_left, model_side, results_left, "Left")
            classify_image_side(photo_path_right, model_side, results_right, "Right")
            classify_image_nozzle(photo_path_nozzle, model_nozzle, results_nozzle, "Nozzle")

            x,y,z = answers(results_left=results_left,results_right=results_right, results_nozzle=results_nozzle)
            conclusion(x,y,z)

        else:
            print("Error capturing from one or more cameras.")
        time.sleep(1)

def classify_image_nozzle(img_path, model, result_list, camera_name):
    results = model.predict(source = img_path, imgsz = 448)
    predicted_class = results[0].names[results[0].probs.top1]
    result_list.append((os.path.basename(img_path), predicted_class))
    print(f"{camera_name} image {os.path.basename(img_path)} classified as {predicted_class}")

def classify_image_side(img_path, model, result_list, camera_name):
    results = model.predict(source=img_path, imgsz = 512)
    predicted_class = results[0].names[results[0].probs.top1]
    result_list.append((os.path.basename(img_path), predicted_class))
    print(f"{camera_name} image {os.path.basename(img_path)} classified as {predicted_class}")


if __name__ == "__main__":
    capture_and_classify_images()
