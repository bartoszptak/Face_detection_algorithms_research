import dlib
import cv2
import os


class DlibModel:
    def __init__(self, size, gpu):

        if gpu:
            dlib.DLIB_USE_CUDA = True
        else:
            dlib.DLIB_USE_CUDA = False

        self.detector = dlib.get_frontal_face_detector()
        self.size = size

    def predict(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)

        def rect_to_bb(rect):
            x = rect.left()
            y = rect.top()
            w = rect.right()
            h = rect.bottom()

            return (x, y, w, h)

        rects = [rect_to_bb(r) for r in rects]

        return rects


class DlibModel2:
    def __init__(self,
                 size,
                 dat_file=os.path.join('models', 'dlib', 'mmod_human_face_detector.dat')):

        if gpu:
            dlib.DLIB_USE_CUDA = True
        else:
            dlib.DLIB_USE_CUDA = False

        self.detector = dlib.cnn_face_detection_model_v1(dat_file)
        self.size = size

    def predict(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)

        def rect_to_bb(rect):
            x = rect.left()
            y = rect.top()
            w = rect.right()
            h = rect.bottom()

            return (x, y, w, h)

        rects = [rect_to_bb(r.rect) for r in rects]

        return rects
