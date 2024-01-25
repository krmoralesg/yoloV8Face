# Face Detection with YOLOv8

### Steps to run Code

- Clone the repository
```
git clone https://github.com/noorkhokhar99/face-detection-yolov8.git
```

- Goto cloned folder
```
cd face-detection-yolov8
```

- Install the ultralytics package
```
pip install ultralytics==8.0.0
```

- Do Tracking with mentioned command below
```
#imagefile
python filename.py
```
# video 
```
yolo task=detect mode=predict model=yolov8n-face.pt conf=0.25 imgsz=1280 line_thickness=1 max_det=1000 source=0
```
credits to pyresearch -->
 I would love your support on Pyresearch: https://youtu.be/_eSArKZBWmE
