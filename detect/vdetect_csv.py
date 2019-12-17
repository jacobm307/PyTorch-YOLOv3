import argparse

import cv2
import mss
import numpy as np
import torch
import time
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from models import *
from utils.utils import *


def prepare_image(img):
    patch = opt.img_size

    img = img[:, :, :3]
    img = img.unsqueeze(0).permute(0, 3, 1, 2) # NHWC -> NCHW
    img *= 1/255.0

    n, c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=0)

    img = F.interpolate(img, size=patch, mode="nearest")

    return img

class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.preload()

    def preload(self):
        try:
            self.next_image = next(self.loader)
            self.next_input = torch.from_numpy(self.next_image)
        except StopIteration:
            self.next_image = None
            self.next_input = None
            return

        if self.stream is None:
            self.next_input = prepare_image(self.next_input.float())
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True).float()
            self.next_input = prepare_image(self.next_input)

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_input is None:
            raise StopIteration

        if self.stream is None:
            self.preload()
            return self.next_image, self.next_input

        image = self.next_image

        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        
        if input is not None:
            input.record_stream(torch.cuda.current_stream())

        self.preload()
        return image, input

class VideoLoader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError()

    def __next__(self):
        if not self.cap.isOpened():
            raise StopIteration()

        ret, image = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration()

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __iter__(self):
        return self

def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w)
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h)
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w)
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h)

    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
    boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
    return boxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="../data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="../config/yolov3-tiny-softball.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="../checkpoints/yolov3_ckpt_285.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="../data/softball.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.9, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.3, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--display", default=False, action='store_true', help="size of each image dimension")
    parser.add_argument("--export", default=True, help="boolean to export to a csv file")
    parser.add_argument("--vid", type=str, default='../UNC2019-04-21GM1.mp4', help="path to the video being used")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path, map_location=device))

    model.eval()  # Set in evaluation mode

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    bbox_colors = []
    for i in np.linspace(0, 1, 10):
        r,g,b,a = cmap(i)
        bbox_colors += [(b*255.0, g*255.0, r*255.0)]
    text_size = 4

    
    result = []

    classes = load_classes(opt.class_path)  # Extracts class labels from file
    data_loader = DataPrefetcher(VideoLoader(opt.vid))
    data_time = time.time()
    for i, (image, input_imgs) in enumerate(data_loader):
        load_time = time.time() - data_time

        thickness = (image.shape[0] + image.shape[1]) // 600

        start_time = time.time()
        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        for box_attr in detections:
            if box_attr is None:
                continue

            labels = []
            detections = rescale_boxes(box_attr, opt.img_size, image.shape[:2])

            print(f'\nFrame: {i: <6} Inference Time: {(time.time() - start_time) * 1000:.2f} + {load_time * 1000:.2f}ms')

            # Rescale boxes to original image
            for dects in detections:
                cx, cy, w, h, conf, cls_conf, cls_pred = [x.item() for x in dects]
                print(f"\t + Label: {classes[int(cls_pred)]: <6} | Obj Conf: {conf:.5f} | Class Conf: {cls_conf:.5f}")
                if cls_conf < 0.6:
                    continue

                result += [(i, int(cls_pred), cx, cy, w, h, conf, cls_conf)]

                if opt.export:
                    org_h, org_w = image.shape[:2]
                    f = open("locations.csv", "a")
                    f.write(str(i+1) + "," + classes[int(cls_pred)] + "," + str(int(cx*org_w)) + "," + str(int(cy*org_h))+ "," + str(w) + "," + str(h) + "\n")
                    f.close()
                
                if opt.display:
                    labels += [result[-1][1:6]]
                    org_h, org_w = image.shape[:2]
                    x1 = int((cx - w / 2) * org_w)
                    y1 = int((cy - h / 2) * org_h)
                    x2 = int((cx + w / 2) * org_w)
                    y2 = int((cy + h / 2) * org_h)
                    color = bbox_colors[int(cls_pred)]
                    label = classes[int(cls_pred)]
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    text_end = x1 + text_width + thickness, y1 - text_height - baseline - thickness // 2

                    cv2.rectangle(image, (x1 - thickness // 2, y1), text_end, color, thickness=cv2.FILLED)
                    cv2.putText(image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        if opt.display:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow('YOLOv3', image)
            key = cv2.waitKey()
            if key == 27 or key == ord('q'):
                break
            if key == ord('c'):
                continue
            if key == ord('s'):
                name = f'../output/{i}'
                cv2.imwrite(name + '.png', image)
                with open(name + '.txt', 'w') as f:
                    for label in labels:
                        print(' '.join([str(x) for x in label]), file=f)
                print(f'Result saved to {name}')

        data_time = time.time()
