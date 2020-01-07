import cv2
import os


class HaarModel:
    def __init__(self,
                 size,
                 gpu,
                 xml_path=os.path.join(
                     'models', 'haarcascade', 'haarcascade_frontalface_default.xml')
                 ):
        if gpu:
            print('[LOGS] CascadeClassifier not use GPU anymore')
            
        self.face_cascade = cv2.CascadeClassifier(xml_path)
        self.size = size

    def predict(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        scale = img.shape[0]/self.size

        gray = cv2.resize(gray, (int(img.shape[1]*scale), int(img.shape[0]*scale)))

        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        faces_new = []
        for ff in faces:
            faces_new.append([int(f/scale) for f in ff])

        print(faces_new)
        return faces_new
