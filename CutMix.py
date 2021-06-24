import numpy as np
import random

from utils import image_resize, xywh2xyxy, xyxy2xywh


def cutmix(image, boxes, r_image, r_boxes):
    """
    Implementation of the CutMix algorithm as described in the Arxiv paper: https://arxiv.org/abs/1905.04899v2
        Parameter:
            image (ndarray): First input image and main image for the cutout application
            boxes (ndarray): numpy array of bounding boxes beloning to first input image
            r_image (ndarray): Second input image that will be used as
            r_boxes (ndarray): numpy array of bounding boxes beloning to first input image

        Returns:
            A new image consisting of merged input images and merged input
    """

    boxes = xywh2xyxy(boxes)
    mixup_image = image.copy()

    # Get image and r_image shapes
    img_h, img_w = image.shape[:2]
    r_img_h, r_img_w = r_image.shape[:2]

    # Check what axis is largest
    if image.shape is not r_image.shape:
        imsize_w = min(img_w, r_img_w)
        imsize_h = min(img_h, r_img_h)
        imsize = min(imsize_w, imsize_h)
        r_image, ratio = image_resize(r_image, imsize_w, imsize_h)
    else:
        imsize = min(image.shape[0], r_image.shape[0])

    # Rescale normalized r_boxes coords, according to ratio of resized r_image
    r_boxes = np.array([[r_box[0], *(r_box[1:] * ratio)] for r_box in r_boxes])

    # Create random mask rectangle
    x1, y1 = [int(random.uniform(imsize * 0.0, imsize * 0.45)) for _ in range(2)]
    x2, y2 = [int(random.uniform(imsize * 0.55, imsize * 1.0)) for _ in range(2)]

    mixup_boxes = r_boxes.copy()
    area = (r_boxes[:, 3] - r_boxes[:, 1]) * (r_boxes[:, 4] - r_boxes[:, 2])

    # Normalize mask coordinates
    nx1 = x1 / img_w
    nx2 = x2 / img_w
    ny1 = y1 / img_h
    ny2 = y2 / img_h

    rx1 = x1 * ratio
    rx2 = x2 * ratio
    ry1 = y1 * ratio
    ry2 = y2 * ratio

    # Clip bounding boxes to
    mixup_boxes[:, [1, 3]] = mixup_boxes[:, [1, 3]].clip(min=rx1, max=rx2)
    mixup_boxes[:, [2, 4]] = mixup_boxes[:, [2, 4]].clip(min=ry1, max=ry2)

    # mixup_boxes = clip_coords_int(mixup_boxes, (rx1, ry1, rx2, ry2))
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(min=nx1, max=nx2)
    boxes[:, [2, 4]] = boxes[:, [2, 4]].clip(min=ny1, max=ny2)

    # mixup_boxes = mixup_boxes.astype(np.int32)

    # cropped w, h, area
    w = mixup_boxes[:, 3] - mixup_boxes[:, 1]
    h = mixup_boxes[:, 4] - mixup_boxes[:, 2]
    area0 = w * h
    ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))

    mixup_boxes = mixup_boxes[np.where((w > 2) & (h > 2) & (area / (area0 + 1e-16) > 0.2) & (ar < 20))]

    # Replace masked rectangle area with same area in r_image
    mixup_image[y1:y2, x1:x2] = r_image[y1:y2, x1:x2]

    boxes = xyxy2xywh(boxes)
    mixup_boxes = np.concatenate((boxes, mixup_boxes), axis=0)

    return mixup_image, mixup_boxes