import cv2

from CutMix import cutmix
from utils import load_labels, plot_bboxes

if __name__ == "__main__":
    img = cv2.imread('test_resources/images/00150.jpg')
    img2 = cv2.imread('test_resources/images/00858.jpg')
    labels2 = 'test_resources/labels/00858.txt'
    labels1 = 'test_resources/labels/00150.txt'

    l1 = load_labels(labels1)
    l2 = load_labels(labels2)

    mix_img, labels = cutmix(img, l1, img2, l2)
    final_img = plot_bboxes(mix_img, labels)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
    cv2.imshow("output", final_img)  # Show image
    cv2.waitKey(0)
