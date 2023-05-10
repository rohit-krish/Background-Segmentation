import cv2
import numpy as np

cap1 = cv2.VideoCapture('../videos/_import_624e6ff9248421.46519419_preview.mp4')
cap2 = cv2.VideoCapture('../videos/outputs4/output_one.mp4')
cap3 = cv2.VideoCapture('../videos/outputs2/output_one.mp4')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('combined.mp4', fourcc, cap1.get(cv2.CAP_PROP_FPS), (1280*3, 720))

ret1, ret2, ret3 = True, True, True

while (ret1 and ret2 and ret3):
    ret1, img1 = cap1.read()
    ret2, img2 = cap2.read()
    ret3, img3 = cap3.read()


    if (ret1 and ret2 and ret3):
        out.write(np.hstack([img1, img2, img3]))

out.release()
