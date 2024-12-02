import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import time
import paho.mqtt.client as mqtt
from datetime import datetime, timedelta

model = YOLO("yolo11n.pt")  # Initialize without specifying device

video_path = 0  # Change to use webcam
cap = cv2.VideoCapture(1)

start_time = time.time()
frame_count = 0
fps = 0.0
last_reset_date = datetime.now()

DwellTime = {}
LastTimeDetected = {}
ClassNames = {}
DwellTimeDetected = {}
TimeStamp = {}

while True:
    success, img = cap.read()
    if not success:
        break

    (H, W) = img.shape[:2]

    # Run YOLO tracking
    results = model.track(img, persist=True, classes=[0, 2, 1, 7, 3, 5])  # person, car, bicycle, truck, motorcycle, bus

    if results and len(results) > 0:
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(box.conf)
                cls = int(box.cls)
                track_id = int(box.id) if box.id is not None else None

                if track_id is not None:

                    class_name = model.names[cls]
                    if class_name == "person":
                        class_name = "Video A"
                    elif class_name == "car":
                        class_name = "Video B"
                    elif class_name == "motorcycle":
                        class_name = "Video C"
                    elif class_name == "truck":
                        class_name = "Video D"
                     
                    centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    # cv2.putText(img, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    cv2.putText(img, f"ID: {track_id}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                    # Draw centroid
                    cv2.circle(img, centroid, 4, (0, 255, 0), -1)
                    cv2.putText(img, f"({centroid[0]}, {centroid[1]})", (centroid[0] + 10, centroid[1] + 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    timestamp = datetime.now()
                    if track_id not in LastTimeDetected:
                        LastTimeDetected[track_id] = timestamp
                        ClassNames[track_id] = class_name
                        DwellTimeDetected[track_id] = 0
                        TimeStamp[track_id] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    elif (timestamp - LastTimeDetected[track_id]).total_seconds() > 0.1:
                        ClassNames[track_id] = class_name
                        DwellTimeDetected[track_id] += 0.1
                        LastTimeDetected[track_id] = timestamp
                        TimeStamp[track_id] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    
                    print(f"DwellTimeDetected: {DwellTimeDetected}")
                    print(f"ClassNames: {ClassNames}")

                    cv2.putText(img, f"{ClassNames[track_id]}", (x1, y1 - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(img, f"ID: {TimeStamp[track_id]}", (x1, y1 - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(img, f"DwellTimeDetected: {DwellTimeDetected[track_id]:.1f}", (x1, y1 - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    # cv2.putText(img, f"{ClassNames[track_id]} DwellTimeDetected: {DwellTimeDetected[track_id]:.1f} TimeStamp: {TimeStamp[track_id]}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    frame_count += 1

    if frame_count % 30 == 0:
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        start_time = time.time()
        frame_count = 0

    cv2.putText(img, f"FPS: {fps:.2f}", (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Image", img)

    with open('DwellTimeDetected.txt', 'w') as file:
        for i in range(1, len(ClassNames) + 1):
            file.write(f"{i}. Video: {list(ClassNames.values())[i-1]}, Timestamp: {list(TimeStamp.values())[i-1]}, LamaTayang: {list(DwellTimeDetected.values())[i-1]:.1f} s\n")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
# out.release()  # Pastikan untuk melepaskan objek penulis video
cv2.destroyAllWindows()
