from sort import *
from multiprocessing import Process
from threading import Thread
from ultralytics import YOLO
import cv2
from PIL import Image
import torch
from telebot_utils.bot import sign_handler
# from threading import Thread
# import time
class CustomProcess(Process):
    def __init__(self,sleep_time):
        Process.__init__(self)
        self.sleep_time = sleep_time    
    def run_thread(self, text):
        thread = Thread(target=sign_handler, args=(text, ))
        thread.start()
        time.sleep(self.sleep_time)
        thread.join()    
    def run(self):
        while True:
            pass
        
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

fire_text = "ALARM, FIRE detected in your camera"
smoke_text = "ALARM, SMOKE detected in your camera"

if __name__ == "__main__":
    
    args = parse_args()

    process = CustomProcess(args.sleep_time)
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

    ret, frame = cap.read()
    classes = ["Fire", "Default", "Smoke"]
    frame_height, frame_width, _ = frame.shape
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            print("No more frame to detect")
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(frame)
        detections = detect_image(pilimg)
        if detections is not None:
            
            tracked_objects = mot_tracker.update(detections)

            for x1, y1, x2, y2, cls_pred, id in tracked_objects:
                x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
                cls = classes[int(cls_pred)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
                # cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60, y1), color, -1)
                cv2.putText(frame, cls+' '+str(int(id)), (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                if cls_pred == 0:
                    process.run_thread(fire_text)
                elif cls_pred == 2:
                    process.run_thread(smoke_text)
                
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("frame", frame)
        
        cv2.waitKey(1)
    print("Done processing video")