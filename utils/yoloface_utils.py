import numpy as np

import cv2
import os


class YolofaceModel:
    def __init__(self,
                 size,
                 cfg=os.path.join('models', 'yoloface', 'yolov3-face.cfg'),
                 weights=os.path.join('models', 'yoloface', 'yolov3-wider_16000.weights')):

        self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
        self.size = size

        def get_outputs_names(net):
            # Get the names of all the layers in the network
            layers_names = net.getLayerNames()

            # Get the names of the output layers, i.e. the layers with unconnected
            # outputs
            return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        self.outputs = get_outputs_names(self.net)

    def predict(self, img):
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (self.size, self.size),
                                     [0, 0, 0], 1, crop=False)

        self.net.setInput(blob)
        outs = self.net.forward(self.outputs)

        final_boxes = self.post_process(img, outs, 0.5, 0.4)

        return final_boxes

    def refined_box(self, left, top, width, height):
        right = left + width
        bottom = top + height

        original_vert_height = bottom - top
        top = int(top + original_vert_height * 0.15)
        bottom = int(bottom - original_vert_height * 0.05)

        margin = ((bottom - top) - (right - left)) // 2
        left = left - margin if (bottom - top - right +
                                 left) % 2 == 0 else left - margin - 1

        right = right + margin

        return left, top, right, bottom

    def post_process(self, frame, outs, conf_threshold, nms_threshold):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only
        # the ones with high confidence scores. Assign the box's class label as the
        # class with the highest score.
        confidences = []
        boxes = []
        final_boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant
        # overlapping boxes with lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                                   nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            final_boxes.append(self.refined_box(left, top, width, height))

        return final_boxes
