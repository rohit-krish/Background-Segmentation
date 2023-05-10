import os
import random
import cv2


class MergeBG:
    def __init__(self, bg_path):
        self.bg_path = bg_path
        self.bgs = os.listdir(bg_path)
        random.shuffle(self.bgs)
        self.idx = 0

    def _set_idx(self, val):
        self.idx = min(max(0, val), len(self.bgs)-1)

    def get(self, image, mask):
        bg = cv2.imread(os.path.join(self.bg_path, self.bgs[self.idx]))
        bg = cv2.resize(bg, (256, 256))

        if image.shape[:2] != (256, 256):
            image = cv2.resize(image, (256, 256))

        if mask.shape[:2] != (256, 256):
            mask = cv2.resize(mask, (256, 256))

        mask_inv = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY_INV)[1]

        bg_masked = cv2.bitwise_and(bg, mask_inv)
        img_masked = cv2.bitwise_and(image, mask)

        res = cv2.add(img_masked, bg_masked)

        self._set_idx(self.idx+1)

        return res


if __name__ == '__main__':
    inst = MergeBG('./data/backgrounds')

    img = cv2.imread('./img.jpg')
    mask = cv2.imread('./mask.jpg')

    inst.get(img, mask)
    inst.get(img, mask)
    inst.get(img, mask)
