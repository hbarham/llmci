


from ultralytics import YOLO
import cv2


model = YOLO('yolov8l.pt')


results = model("./trial/image/3bg.webp", show=True)

cv2.waitKey(0)


