import cv2
import os


class DNNModel:
    def __init__(self,
                 size,
                 gpu,
                 model=os.path.join('models', 'dnn', 'opencv_face_detector_uint8.pb'),
                 proto=os.path.join(
                     'models', 'dnn', 'opencv_face_detector.pbtxt')):

        self.size = size
        self.net = cv2.dnn.readNetFromTensorflow(model, proto)

        if gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def predict(self, img):
        frameHeight = img.shape[0]
        frameWidth = img.shape[1]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], True, False)

        self.net.setInput(blob)
        detections = self.net.forward()

        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.4:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append((x1, y1, x2, y2))

        return bboxes