from ultralytics import YOLO
import cv2
model_path = 'best.pt'
image_path = 'test.jpg'
model = YOLO(model_path)

results = model.predict(source=image_path, conf=0.25)
# print(results)
print(results[0].boxes.xyxy)
print(results[0].boxes.cls)

img = cv2.imread(image_path)
x1 = int(results[0].boxes.xyxy[0][0].detach().cpu().numpy())
y1 = int(results[0].boxes.xyxy[0][1].detach().cpu().numpy())
x2 = int(results[0].boxes.xyxy[0][2].detach().cpu().numpy())
y2 = int(results[0].boxes.xyxy[0][3].detach().cpu().numpy())

img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
import pdb; pdb.set_trace()
cv2.imwrite('test_out.jpg', img)
