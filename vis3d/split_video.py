import cv2

cap = cv2.VideoCapture("nus_project.mp4")
i=0
writer = None
while True:
    flag,_ = cap.read()

    if not flag:break
    i += 1
    if i % 5 == 0:
        if writer is None:
            writer = cv2.VideoWriter(f"nus_project_111.mp4",
                                     cv2.VideoWriter.fourcc(*"mp4v"),
                                     10.,
                                     (_.shape[1], _.shape[0]))
        writer.write(_)
print(i)
