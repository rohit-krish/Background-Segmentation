import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from tqdm import tqdm
import time


class SegIt:
    def __init__(self, model_path, bg_path=None):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path).to(self.device)
        self.model.eval()

        self.transform = A.Compose([
            A.Resize(height=256, width=256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        self.input_img = None
        self.pred_mask = None

        self.res_frames = []

        self.bg = None if bg_path == None else cv2.imread(bg_path)
        self._got_img_h_and_w = False

    def do(self, img):
        h, w, *_ = img.shape
        self.input_img = img

        imgX = cv2.resize(img, (256, 256))
        imgX = cv2.cvtColor(imgX, cv2.COLOR_BGR2RGB)
        imgX = self.transform(image=imgX)['image'].unsqueeze(0)
        imgX = (imgX - imgX.min()) / (imgX.max() - imgX.min())

        imgX = imgX.to(self.device)

        with torch.no_grad():
            pred_mask = (torch.sigmoid(self.model(imgX)[
                         'out']) > .5).cpu().float().numpy()

        pred_mask = pred_mask.squeeze().astype('uint8') * 255

        pred_mask = cv2.cvtColor(cv2.resize(
            pred_mask, (w, h)), cv2.COLOR_GRAY2BGR)

        self.pred_mask = pred_mask

        if (self._got_img_h_and_w == False) and (self.bg is not None):
            self.bg = cv2.resize(self.bg, (w, h))
            self._got_img_h_and_w = True

        return pred_mask

    def merge(self):
        if (self.input_img is None) or (self.pred_mask is None):
            raise Exception('Perform the segmentation first!')

        if self.bg is None:
            res = cv2.bitwise_and(self.input_img, self.pred_mask)
        else:
            mask_inv = cv2.threshold(
                self.pred_mask, 180, 255, cv2.THRESH_BINARY_INV)[1]
            bg_masked = cv2.bitwise_and(self.bg, mask_inv)
            img_masked = cv2.bitwise_and(self.input_img, self.pred_mask)
            res = cv2.add(img_masked, bg_masked)

        self.res_frames.append(res)

        return res

    def make_video(self, fps, frame_shape, file_name):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(file_name+'.mp4', fourcc, fps, frame_shape)

        for frame in tqdm(self.res_frames):
            out.write(frame)

        out.release()

        self.res_frames = []


def trim_video(input_file, output_file, start_sec, end_sec):
    # Open the input video file
    cap = cv2.VideoCapture(input_file)

    # Get the frame rate and total number of frames in the input video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the start and end frame numbers based on the input start and end times
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    # Make sure the start and end frames are within the bounds of the video
    start_frame = max(0, start_frame)
    end_frame = min(total_frames - 1, end_frame)

    # Set the starting position of the video to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Create a VideoWriter object to write the output video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Loop through the frames of the input video and write the frames to the output video
    for i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            break

    # Release the video files
    cap.release()
    out.release()


class GetFPS:
    def __init__(self) -> None:
        self.prev_time = 0
        self.curr_time = 0

    def get(self):
        self.curr_time = time.time()
        fps = 1/(self.curr_time-self.prev_time)
        self.prev_time = self.curr_time
        return int(fps)

    def draw_in_img(self, img, scale=1):
        cv2.rectangle(img, (0, 0), (int(200*scale), int(50*scale)),
                      (100, 46, 21), cv2.FILLED)
        cv2.putText(
            img, f'FPS: {self.get()}', (int(10*scale), int(40*scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5*scale, (255, 255, 255), int(2*scale)
        )
        return img


def _resize_and_fill_gaps(matrix, scale, label_height):
    # height and width of the first image
    height, width, *_ = matrix[0][0].shape
    height = int(height * scale)
    width = int(width * scale)

    n_rows = len(matrix)
    n_cols = 0

    for row in matrix:
        if len(row) > n_cols:
            n_cols = len(row)

    result = np.zeros(
        (n_rows, n_cols, height+label_height, width, 3), dtype=np.uint8)

    for r_idx, row in enumerate(matrix):
        for c_idx, img in enumerate(row):
            img = np.squeeze(img)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            img = cv2.resize(img, (width, height))
            text_place = np.zeros((label_height, width, 3))

            result[r_idx, c_idx] = np.vstack((img, text_place))

    return result


def stackIt(img_matrix, label_matrix=None,  img_scale=1, label_height=30, **kwargs):
    label_height = 0 if label_matrix == None else label_height
    img_matrix = _resize_and_fill_gaps(img_matrix, img_scale, label_height)

    # putting the labels in each images
    if label_matrix:
        for img_row, label_row in zip(img_matrix, label_matrix):
            for image, label in zip(img_row, label_row):
                h, *_ = image.shape
                cv2.putText(image, label, (10, h-10), **kwargs)

    row_images = []
    for row in img_matrix:
        row_images.append(np.hstack(tuple([*row])))

    return np.vstack(tuple([*row_images]))
