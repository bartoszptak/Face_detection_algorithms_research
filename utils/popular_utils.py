import cv2
import numpy as np


def draw_predict(img, preds):
    for left, top, right, bottom in preds:
        cv2.rectangle(img, (left, top), (right, bottom), (0,0,255), 2)


def quantize_colors(img, colors):
    """Reduce the number of colors in the image."""
    img = img.astype('float')
    return np.round(img/255*colors)/colors*255

def blur_predict(img, preds):
    return NotImplementedError
    # for left, top, right, bottom in preds:
    #     roi = img[top:bottom, left:right]
    #     roi = quantize_colors(roi, 4)
    #     roi = cv2.GaussianBlur(roi, (7,7), cv2.BORDER_REFLECT)
    #     img[top:bottom, left:right] = roi

def make_benchmark(net, size, data):
    return NotImplementedError

    acc, fps = 0, 0

    return acc, fps