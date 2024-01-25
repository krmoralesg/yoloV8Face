from ultralytics import YOLO
#from ultralytics.yolo.v8.detect.predict import DetectionPredictor

import cv2
import os

# Path to the input folder containing images
input_folder = "tars"
# Path to the output folder to save cropped images
output_folder = "results"

# Instantiate your YOLO model (assuming you have already defined the YOLO class)
model = YOLO("yolov8n-face.pt")

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate over all image files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        # Full path to the input image
        img_path = os.path.join(input_folder, filename)

        # Run YOLO on the current image
        results = model(img_path)
        boxes = results[0].boxes

        # Read the image
        img = cv2.imread(img_path)

        # Use a counter variable
        index = 1

        # Iterate over bounding boxes and save cropped images
        for box in boxes:
            top_left_x = int(box.xyxy.tolist()[0][0])
            top_left_y = int(box.xyxy.tolist()[0][1])
            bottom_right_x = int(box.xyxy.tolist()[0][2])
            bottom_right_y = int(box.xyxy.tolist()[0][3])

            print(f"Top-left coordinates: ({top_left_x}, {top_left_y})")
            print(f"Bottom-right coordinates: ({bottom_right_x}, {bottom_right_y})")

            # Crop the image
            cropped_image = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            # Save the cropped image with an index in the output folder
            output_filename = f"{output_folder}/{filename}"
            cv2.imwrite(output_filename, cropped_image)

            # Increment the index
            index += 1