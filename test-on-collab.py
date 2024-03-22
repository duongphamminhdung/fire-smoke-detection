from sort import *

from ultralytics import YOLO
import cv2
from PIL import Image
import torch
from google.colab.patches import cv2_imshow
import os
import time
from multiprocessing import Process
from multiprocessing import Pool
from threading import Thread
from ultralytics import YOLO
# from telebot_utils.bot import sign_handler

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

args = parse_args()

name = args.name
model_path = args.model_path
video_path = args.video_path

model = YOLO(model_path)
mot_tracker = Sort(max_age=args.max_age, 
                    min_hits=args.min_hits,
                    iou_threshold=args.iou_threshold)
if video_path.isnumeric():
    cap = cv2.VideoCapture(int(video_path))
else:
    cap = cv2.VideoCapture(video_path)
print(video_path)
ret, frame = cap.read()
classes = ["Fire", "Default", "Smoke"]
frame_height, frame_width, _ = frame.shape
frame_count = 0
t = 0

os.makedirs('test', exist_ok = True)
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 46, (frame_width,frame_height))

# class CustomProcess(Process):
#     def __init__(self,sleep_time):
#         Process.__init__(self)
#         self.sleep_time = sleep_time    
#     def run_thread(self, text):
#         thread = Thread(target=sign_handler, args=(text, ))
#         thread.start()
#         time.sleep(self.sleep_time)
#         thread.join()    
#     def run(self):
#         while True:
#             pass
def detect_and_update(frame, out):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)
    if detections is not None:
        frame_count = 0
        tracked_objects = mot_tracker.update(detections)
        # import pdb; pdb.set_trace()

        for x1, y1, x2, y2, id, cls_pred in tracked_objects:
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            cls = classes[int(cls_pred)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
            # cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60, y1), color, -1)
            cv2.putText(frame, cls+" "+ str(int(id)), (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    end = time.time()
    fps = 1/(end-start)
    cv2.putText(frame, "fps: "+str(fps)[:3], (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    out.write(frame)     
           
fire_text = "ALARM, FIRE detected in your camera"
smoke_text = "ALARM, SMOKE detected in your camera"
processes = []
if __name__ == '__main__':
    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        t += 1
        frame_count += 1
        if not ret:
            out.release()
            print("No more frame to detect")
            break
        if len(processes) >= 3:
            processes[0].join()
            processes.pop(0)
        processes.append(Process(target=detect_and_update, args=(frame, out, )))
        processes[-1].start()

    print("Done processing video")
    out.release()