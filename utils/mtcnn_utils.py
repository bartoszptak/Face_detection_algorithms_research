from mtcnn import MTCNN
import cv2


class MtcnnModel:
    def __init__(self, size):
        self.detector = MTCNN()

    def predict(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pred = self.detector.detect_faces(img)

        boxes = []
        for p in pred:
            bounding_box = p['box']
            boxes.append((bounding_box[0], bounding_box[1], bounding_box[0] +
                         bounding_box[2], bounding_box[1] + bounding_box[3]))

        return boxes
