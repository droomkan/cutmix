import numpy as np
import random

from utils import image_resize, xywh2xyxy, xyxy2xywh, get_absolute_coords, get_relative_coords


def cutmix(image, boxes, r_image, r_boxes):
    """
    Implementation of the CutMix algorithm as described in the Arxiv paper: https://arxiv.org/abs/1905.04899v2
    This implementation will not be completely accurate as it tries to apply cutmix for an object-detection context.
    A couple of problems are posed related to different dimensions of input images and translation of ground truth
    bounding boxes.

    This implementation also does not use beta-distribution, but simply generates a random mask bounding mask based on
    the smallest dimension between the two input images.

        Parameter:
            image (ndarray): First input image and target image for the cutout application
            boxes (ndarray): numpy array of bounding boxes beloning to first input image
            r_image (ndarray): Second input image that will be used as cutout area
            r_boxes (ndarray): numpy array of bounding boxes belonging to first input image

        Returns:
            A new image consisting of merged input images and merged input
    """

    # Convert bounding boxes from original yolov4 format [cls, cx, cy, w, h] to [cls, x1, y1, x2, y2]
    boxes = xywh2xyxy(boxes)
    r_boxes = xywh2xyxy(r_boxes)

    # Copy target image
    target_image = image.copy()

    # Get image sizes
    img_size = image.size
    r_img_size = r_image.size

    # Get image and r_image shapes
    img_h, img_w = image.shape[:2]
    r_img_h, r_img_w = r_image.shape[:2]
    r_boxes = get_absolute_coords(r_boxes, r_img_w, r_img_h)

    # Determine whether second input image need to be up-scaled or down-scaled based on target input image
    if image.shape is not r_image.shape or img_size is not r_img_size:
        r_image, ratio = image_resize(r_image, img_w, img_h)

        if img_size >= r_img_size:
            r_img_h, r_img_w = r_image.shape[:2]
            imsize = min(r_img_h, r_img_w)
        else:
            imsize = min(img_w, img_h)

        # Rescale normalized r_boxes coords, according to ratio of resized r_image
        r_boxes[:, 1:] = r_boxes[:, 1:] * ratio

    else:
        imsize = min(img_w, img_h)

    # Create random regional dropout bounding box based on smallest image
    x1, y1 = [int(random.uniform(imsize * 0.0, imsize * 0.45)) for _ in range(2)]
    x2, y2 = [int(random.uniform(imsize * 0.55, imsize * 1)) for _ in range(2)]

    dropout_box = [x1, y1, x2, y2]

    # Remove bounding boxes  from mixup_boxes that are not inside the random regional dropout area
    index = [i for i, box in enumerate(r_boxes) if bbox_ioa(box[1:], dropout_box) <= 0.01]
    r_boxes = np.delete(r_boxes, index, axis=0)

    mixup_boxes = r_boxes.copy()
    # Determine which boxes fall outside dropout region and clip them to the mask outer limits if there is overlap
    # If bounding box is completely overlapped by regional dropout mask then remove it

    # Clip sample image bounding boxes to the inside of the generated mask region
    mixup_boxes[:, [1, 3]] = mixup_boxes[:, [1, 3]].clip(min=x1, max=x2)
    mixup_boxes[:, [2, 4]] = mixup_boxes[:, [2, 4]].clip(min=y1, max=y2)

    # Translate normalized boxes to absolute coords
    boxes = get_absolute_coords(boxes, img_w, img_h)

    # For all boxes in target image check if they overlap with dropout region
    # We need to adjust the boxes so that they will clip to the outside bounds of the dropout region
    for i, box in enumerate(boxes[:, 1:]):
        # First check if bounding box is even inside dropout-region, if not skip entire iteration
        iou = bbox_ioa(box, dropout_box)
        if iou > 0.01:
            # Check if box completely overlapped. If so, remove it from boxes array
            if is_box_inside(box, dropout_box):
                boxes = np.delete(boxes, i, axis=0)
            else:
                box[:] = clip_outer_to_inner(box, dropout_box)[:]

    # Replace masked rectangle area from selected region of r_image with same area in target image
    target_image[y1:y2, x1:x2] = r_image[y1:y2, x1:x2]

    # Translate r_boxes coords to relative coordinates on target image
    boxes = get_relative_coords(boxes, img_w, img_h)
    mixup_boxes = get_relative_coords(mixup_boxes, img_w, img_h)

    boxes = xyxy2xywh(boxes)
    mixup_boxes = xyxy2xywh(mixup_boxes)

    # Merge all bounding boxes to one array
    mixup_boxes = np.concatenate((boxes, mixup_boxes), axis=0)

    return target_image, mixup_boxes


def bbox_ioa(box1, box2):
    # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

    # Intersection over box2 area
    return inter_area / box2_area


def is_box_inside(ibox, bbox):
    """
    Checks if inner box falls inside bounding box coordinates based on x1, y1, x2, y2 coordinates.
    :param ibox: (array) Inner box to check if it inside supposed outer box
    :param bbox: (array) Bounding box
    :return: (boolean)
    """

    if bbox[0] <= ibox[0] and bbox[1] <= ibox[1]:
        if bbox[2] >= ibox[2] and bbox[3] >= ibox[3]:
            return True
    else:
        return False


def get_box_area(box):
    return (box[3] - box[1]) * (box[2] - box[0])


def clip_outer_to_inner(outer_box, inner_box):
    """
    Will clip the coords of outer box, overlapping with inner box
    :param outer_box:
    :param inner_box:
    :return:
    """
    # If there is no complete overlap by dropout region, figure out whether the overlap happens at
    # left, top right or bottom of dropout region.
    bx1, by1, bx2, by2 = outer_box[0], outer_box[1], outer_box[2], outer_box[3]
    dx1, dy1, dx2, dy2 = inner_box[0], inner_box[1], inner_box[2], inner_box[3]

    candidate_boxes = []

    # Check overlap with left bound
    if bx1 < dx1 < bx2:
        candidate_box = [bx1, by1, dx1, by2]
        candidate_boxes.append(candidate_box)
        # box[2] = dx1

    # Check overlap with top bound
    if by1 < dy1 < by2:
        candidate_box = [bx1, by1, bx2, dy1]
        candidate_boxes.append(candidate_box)
        # box[3] = dy1

    # Check overlap with right bound
    if bx1 < dx2 < bx2:
        candidate_box = [dx2, by1, bx2, by2]
        candidate_boxes.append(candidate_box)
        # box[0] = dx2

    # Check overlap with lower bound
    if by1 < dy2 < by2:
        candidate_box = [bx1, dy2, bx2, by2]
        candidate_boxes.append(candidate_box)
        # box[1] = dy2

    if len(candidate_boxes) == 1:
        new_box = np.array(candidate_boxes[0])
        outer_box[:] = new_box[:]

    elif len(candidate_boxes) > 1:
        max_idx = 0
        max_area = 0

        for i, candidate in enumerate(candidate_boxes):
            box_area = get_box_area(candidate)
            if box_area > max_area:
                max_area = box_area
                max_idx = i
        outer_box[:] = np.array(candidate_boxes[max_idx])[:]

    return outer_box
