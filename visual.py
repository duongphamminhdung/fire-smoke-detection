from sort import *

from ultralytics import YOLO
import cv2
from PIL import Image
import torch

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

def detector(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)
    if detections is not None:
        
        tracked_objects = mot_tracker.update(detections)

        for x1, y1, x2, y2, cls_pred in tracked_objects:
            # box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            # box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            # y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            # x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            cls = classes[int(cls_pred)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
            # cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60, y1), color, -1)
            cv2.putText(frame, cls, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

args = parse_args()

name = args.name
model_path = args.model_path
video_path = args.video_path

total_time = 0.0
total_frames = 0

model = YOLO(model_path)
mot_tracker = Sort(max_age=args.max_age, 
                    min_hits=args.min_hits,
                    iou_threshold=args.iou_threshold)
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
classes = ["Fire", "default", "Smoke"]
frame_height, frame_width, _ = frame.shape
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 46, (frame_width,frame_height))
print("Processing Video...")
while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    out.release()
    break
  output = detector(frame)
  out.write(output)
out.release()
print("Done processing video")