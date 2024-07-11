import sys
import cv2
import numpy as np
import os
import torch
import easyocr
from pathlib import Path 
from detect import detect
from models.experimental import attempt_load
from src.char_classification.model import CNN_Model
import DataBase_generator_14sept
dbdata = DataBase_generator_14sept.dbconnect("localhost", "root", '', "license_plate", "license_plates")

Min_char = 0.01
Max_char = 0.09
CHAR_CLASSIFICATION_WEIGHTS = 'weight.h5'
save_folder_crop = 'crop'
save_folder_detected = 'LP_detected_images'
save_folder_videos = 'processed_videos'

# Create folders if they don't exist
os.makedirs(save_folder_crop, exist_ok=True)
os.makedirs(save_folder_detected, exist_ok=True)
os.makedirs(save_folder_videos, exist_ok=True)

# Load character classification model
model_char = CNN_Model(trainable=False).model
model_char.load_weights(CHAR_CLASSIFICATION_WEIGHTS)

# Check for GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

import datetime


def process_image(img_path):
    source_img = cv2.imread(img_path)
    pred, LP_detected_img = detect(model_LP, source_img, device, imgsz=640)
    filename = os.path.splitext(os.path.basename(img_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    c = 0
    for *xyxy, conf, cls in reversed(pred):
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        LP_cropped = source_img[y1:y2, x1:x2]
        cropped_image_path = os.path.join(save_folder_crop, f'{filename}_{timestamp}_{c}.jpg')

        # Convert LP_cropped to grayscale
        LP_cropped_gray = cv2.cvtColor(LP_cropped, cv2.COLOR_BGR2GRAY)

        # Apply EasyOCR on the grayscale cropped image
        cropped_results = reader.readtext(LP_cropped_gray, detail=0)
        cropped_label = ' '.join(cropped_results)
        print(f'Cropped License Plate Text: {cropped_label}')
        
        dbdata.add_dbdata(cropped_label)

        cv2.imwrite(cropped_image_path, LP_cropped_gray)
        
        cv2.namedWindow('LP_cropped', cv2.WINDOW_NORMAL)
        cv2.imshow(f'LP_cropped', LP_cropped_gray)
        
        strFinalString = ' '.join(cropped_results)  # Corrected variable name here
        cv2.rectangle(LP_detected_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{strFinalString}'
        cv2.putText(LP_detected_img, label, (x1, y1 - 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        
        detected_image_path = os.path.join(save_folder_detected, f'{filename}_{timestamp}_{c}.jpg')
        cv2.imwrite(detected_image_path, LP_detected_img)
        
        print(label)
        c += 1
        cv2.namedWindow('final_result', cv2.WINDOW_NORMAL)
        cv2.imshow('final_result', cv2.resize(LP_detected_img, dsize=None, fx=1, fy=1))
        cv2.waitKey(0)
        
        # Clear previous annotations for the next iteration
        LP_detected_img = source_img.copy()
        
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    video_name = Path(video_path).stem
    save_dir = Path(save_folder_videos)
    counter = 1
    out_video_path = save_dir / f'processed_{video_name}_{counter}.mp4'
    while out_video_path.exists():
        counter += 1
        out_video_path = save_dir / f'processed_{video_name}_{counter}.mp4'
    
    out_video = cv2.VideoWriter(str(out_video_path), fourcc, frame_rate, (frame_width, frame_height))

    filename = os.path.splitext(os.path.basename(video_path))[0]  # Define filename here

    while cap.isOpened():
        ret, source_img = cap.read()
        if not ret:
            break
        pred, LP_detected_img = detect(model_LP, source_img, device, imgsz=640)
        c = 0
        for *xyxy, conf, cls in reversed(pred):
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            LP_cropped = source_img[y1:y2, x1:x2]
            results = reader.readtext(LP_cropped, detail=0)
            strFinalString = ' '.join(results)
            cv2.rectangle(LP_detected_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{strFinalString}'
            cv2.putText(LP_detected_img, label, (x1, y1 - 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('final_result', cv2.resize(LP_detected_img, dsize=None, fx=1, fy=1))  # Show processed frame
            out_video.write(LP_detected_img)  # Write the frame to the output video
            cv2.waitKey(1)  # Wait for a key press to proceed
            print(label)
            
            # Insert text into the database
            dbdata.add_dbdata(label)

            c += 1

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()  # Close all OpenCV windows after processing

best_pt_path = sys.argv[2]
reader = easyocr.Reader(['en'], gpu=True)  # Change languages and GPU usage as needed
model_LP = attempt_load(best_pt_path, map_location=device)

# Check if the input is an image or a video
input_path = sys.argv[4]
if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
    process_image(input_path)
elif input_path.lower().endswith(('.avi', '.mp4', '.mkv', '.webm')):
    process_video(input_path)
else:
    print("Unsupported file format. Please provide an image (PNG, JPG, etc.) or a video (AVI, MP4, etc.).")
