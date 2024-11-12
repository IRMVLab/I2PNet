import cv2

cap = cv2.VideoCapture("kitti_project.mp4")
i=0
while True:
    flag,_ = cap.read()
    if not flag:break
    i += 1
print(i)
