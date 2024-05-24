from ultralytics import YOLO
import cv2 

path = "IMG-0ed4dbcda15e51cab847c0c899a51411-V.jpg"
img  = cv2.imread(path)

width ,height = 800 , 600
imgResize = cv2.resize(img,(width,height))

model = YOLO('../yolov8m.pt')
result = model(imgResize, show=True)
cv2.waitKey(0)