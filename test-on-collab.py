from ultralytics import YOLO
import cv2
from PIL import Image
import torch
from google.colab.patches import cv2_imshow
import os
# from threading import Thread
# import time

def detect_image(img):
    with torch.no_grad():
        results = model.predict(img, conf=0.25)
        bbox_xyxy = np.array([res.cpu().numpy().boxes.xyxy for res in results])
        
        if len(bbox_xyxy[0]) == 0:
            return None
        
        cls_conf = np.array([res.cpu().numpy().boxes.conf for res in results])
        cls_ids = np.array([res.cpu().numpy().boxes.cls for res in results])
        # import pdb; pdb.set_trace()
        dets = np.zeros((len(bbox_xyxy), 6))
        for i in range(len(bbox_xyxy)):
            dets[i] = bbox_xyxy[i][0][0], bbox_xyxy[i][0][1], bbox_xyxy[i][0][2], bbox_xyxy[i][0][3], cls_conf[i][0], cls_ids[i][0]

    return dets


model = YOLO('best.pt')
video_path = 'test.mp4'
if video_path.isnumeric():
    cap = cv2.VideoCapture(int(video_path))
else:
    cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
classes = ["Fire", "Default", "Smoke"]
frame_height, frame_width, _ = frame.shape
frame_count = 0

os.makedirs('test', exist_ok = True)
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 46, (frame_width,frame_height))
while cap.isOpened():
    ret, frame = cap.read()
    frame_count += 1
    if not ret:
        out.release()
        print("No more frame to detect")
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)
    if detections is not None:
        frame_count = 0
        # import pdb; pdb.set_trace()

        for x1, y1, x2, y2, conf, cls_pred in detections:
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            cls = classes[int(cls_pred)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
            # cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60, y1), color, -1)
            cv2.putText(frame, cls+" "+ str(conf), (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite("test/frame"+str(frame_count)+'.jpg', frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame)
            # import pdb; pdb.set_trace()
print("Done processing video")
out.release()