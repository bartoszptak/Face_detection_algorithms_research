import cv2
import os


class HaarModel:
    def __init__(self,
                 size,
                 xml_path=os.path.join(
                     'models', 'haarcascade', 'haarcascade_frontalface_default.xml')
                 ):
        self.face_cascade = cv2.CascadeClassifier(xml_path)

    def predict(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        return faces
