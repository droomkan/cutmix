import random

import cv2
import numpy as np


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized, r


def letterbox(img, new_shape, color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def plot_bboxes(img, targets):
    names = ['Working glove', 'T-shaped pneumatic connector', 'Ball bearing']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    for i, det in enumerate(targets):  # detections per image
        cls = int(det[0])
        xyxy = det[1:]  # x1, y1 are center coordinates for boudning box
        label = names[cls]
        plot_one_box(xyxy, img, label=label, color=colors[cls], line_thickness=3)

    return img


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    w, h, _ = img.shape

    tl = line_thickness or round(0.002 * (w + h) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]

    bbox_w = int(x[2] * h)
    bbox_h = int(x[3] * w)

    x1, y1 = int(x[0] * h), int(x[1] * w)

    c1 = int(x1 - (bbox_w / 2)), int(y1 - (bbox_h / 2))
    c2 = c1[0] + bbox_w, c1[1] + bbox_h

    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def xywh2xyxy(boxes):
    """
    :param boxes: Bounding boxes based in YOLOv4 format center_x, center_y, w, h
    :return: Adjusted bounding boxes of format x1, y1, x2, y2
    """
    c_boxes = boxes.copy()

    for box in c_boxes:
        w = box[3]
        h = box[4]
        box[1] = box[1] - (w / 2)
        box[2] = box[2] - (h / 2)
        box[3] = box[1] + w
        box[4] = box[2] + h

    return c_boxes


def xyxy2xywh(boxes):
    c_boxes = boxes.copy()

    for box in c_boxes:
        w = box[3] - box[1]
        h = box[4] - box[2]
        box[1] = box[1] + (w / 2)  # center x
        box[2] = box[2] + (h / 2)  # center y
        box[3] = w
        box[4] = h

    return c_boxes


def load_labels(labels_file):
    with open(labels_file, 'r') as f:
        labels = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)  # labels
    return labels


def get_absolute_coords(boxes, img_w, img_h):
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * img_w
    boxes[:, [2, 4]] = boxes[:, [2, 4]] * img_h
    return boxes


def get_relative_coords(boxes, img_w, img_h):
    boxes[:, [1, 3]] = boxes[:, [1, 3]] / img_w
    boxes[:, [2, 4]] = boxes[:, [2, 4]] / img_h
    return boxes


    