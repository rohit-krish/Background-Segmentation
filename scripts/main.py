import cv2
from utils import stackIt, SegIt, GetFPS

segment = SegIt('../models/deeplabv3_bg_seg_model_v2.pth', '../data/backgrounds/pic_001[Creation date].jpg')
fps = GetFPS()
cap = cv2.VideoCapture('../videos/_import_624e6ff9248421.46519419_preview.mp4')

while cap.isOpened():
    _, img = cap.read()

    pred_mask = segment.do(img)
    res = segment.merge()
    fps.draw_in_img(res, 2)

    cv2.imshow('Background Removal', stackIt([[img, pred_mask, res]], img_scale=.5))

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()

h, w, *c = img.shape

segment.make_video(cap.get(cv2.CAP_PROP_FPS), (w, h), 'output')
