---
sidebar_label: YOLO (You Only Look Once)
title: Let's get started with YOLO
---

**YOLO (You Only Look Once)** is a popular state-of-the-art object detection system that has revolutionized the field of computer vision. It was first introduced in 2015 by Joseph Redmon, Ali Farhadi, and other researchers from the University of Washington and Allen Institute for AI. YOLO is a deep learning algorithm that can detect and classify objects in real-time images and videos.

One of the key features of YOLO is its incredible speed. It can process images at 45 Frames Per Second (FPS), making it one of the fastest object detection systems available. This is especially important for applications such as autonomous driving and surveillance, where real-time detection is critical.

![speed_of_yolo_comparition](https://res.cloudinary.com/dyd911kmh/image/upload/v1664382693/YOLO_Speed_compared_to_other_state_of_the_art_object_detectors_9c11b62189.png)

`source:` [datacamp](https://www.datacamp.com/blog/yolo-object-detection-explained)

## YOLO Architecture
The architecture of YOLO is very similar to [GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf), another popular deep learning model. YOLO has 24 convolutional layers, four max-pooling layers, and two fully connected layers. The first few layers of the network are used for extracting low-level features from the input image, while the later layers are used for detecting and classifying objects.

<p align="center">
  <img src="https://res.cloudinary.com/dyd911kmh/image/upload/v1664382694/YOLO_Architecture_from_the_original_paper_ff4e5383c0.png" alt="googlenet"/>
</p>

`source:` [datacamp](https://www.datacamp.com/blog/yolo-object-detection-explained)

## YOLO Algorithm
The YOLO algorithm works by dividing the input image into a grid of cells and predicting bounding boxes and class probabilities for each cell. Each bounding box is represented by four values: the x and y coordinates of the center of the box, its width, and its height. The class probabilities represent the likelihood that the bounding box contains an object of a particular class.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:640/0*aIHq0P9eySmRPVxb" alt="bounding boxes"/>
</p>

`source:` [Analytics Vidhya](https://medium.com/analytics-vidhya/yolo-explained-5b6f4564f31)


One of the advantages of YOLO over other object detection systems is its ability to detect multiple objects in a single image. This is achieved by using a technique called non-maximum suppression, which eliminates redundant bounding boxes and keeps only the most probable ones.

Another advantage of YOLO is its accuracy. In benchmark tests, YOLO has been shown to achieve state-of-the-art performance on several object detection datasets, including PASCAL VOC and COCO. However, the accuracy of YOLO can be affected by factors such as image resolution and object size.

Let's dive into getting some hands on experience with yolo.

## Object detection with a webcam (python)

We'll start by install `ultralytics`, a python package for YOLO model.

` pip install ultralytics` should install most of the required dependencies.

The webcam detection script is shown below.
```python
from ultralytics import YOLO
import cv2
import math 

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike","aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow("Image:", img)
    cv2.waitKey(1)
```
[result](https://media.licdn.com/dms/image/C4D22AQHgEf2UsFaFdQ/feedshare-shrink_800/0/1676341840900?e=1682553600&v=beta&t=Pe3M0d-69m_5DAwjNaDAWkJolNVVnxOR4rARrDUamI8)

👉 [**Result webcam detection video**](https://www.linkedin.com/feed/update/urn:li:activity:7031087311996002304/)


