import cv2
from ultralytics import YOLO
import numpy as np
import torch

# print(torch.backends.mps.is_available())

# cap = cv2.VideoCapture("dogs.mp4")
cap = cv2.VideoCapture("t.mp4")
model = YOLO("yolov8m.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device="mps")
    result = results[0]
    bboxes = result.boxes.xyxy

    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")

    print()

    for cls, bbox in zip(classes, bboxes):
        print(result.names[cls.item()])
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            result.names[cls.item()],
            (x, y - 5),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 255),
            2,
        )
    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
