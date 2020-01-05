import os
import numpy as np

import cv2
import click

from utils.popular_utils import draw_predict, blur_predict, make_benchmark



def select_model(model, size):
    if model == 'haar':
        print('[LOGS] Load frontal face cascade classifier from OpenCV')
        from utils.facecascade_utils import HaarModel
        model = HaarModel(size)
    elif model == 'dlib':
        print('[LOGS] Load frontal face detector from dlib')
        from utils.dlib_utils import DlibModel
        model = DlibModel(size)
    elif model == 'dlib2':
        print('[LOGS] Load CNN face detection model from dlib')
        from utils.dlib_utils import DlibModel2
        model = DlibModel2(size)
    elif model == 'yoloface':
        print('[LOGS] Load yoloface model')
        from utils.yoloface_utils import YolofaceModel
        model = YolofaceModel(size)
    elif model == 'mtcnn':
        print('[LOGS] Load face MTCNN model')
        from utils.mtcnn_utils import MtcnnModel
        model = MtcnnModel(size)
    elif model == 'dnn':
        print('[LOGS] Load DNN Face Detector')
        from utils.dnn_utils import DNNModel
        model = DNNModel(size)

    else:
        print('[LOGS] Model option not supported')
        print('[LOGS] Exit')
        exit(0)

    return model


@click.command()
@click.option('--model', default=None, help='Model type [haar, dlib, yoloface, mtcnn]', required=True)
@click.option('--size', default=416, help='Image size', required=True)
@click.option('--mode', default=None, help='Model type [sample, bench]', required=True)
@click.option('--path', default='data/people-brasil-guys-avpaulista-109919.jpg', help='samle: path to image, bench: path do data dir', required=True)
@click.option('--face', default='silent', help='Model type [draw, blur, silent]', required=True)
def main(model, size, mode, path, face):
    size = int(size)
    net = select_model(model, size)

    if mode == 'sample':
        img = cv2.imread(path)
        preds = net.predict(img)

        if face=='draw':
            draw_predict(img, preds)
        elif face=='blur':
            blur_predict(img, preds)

        cv2.imshow('Result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif mode == 'bench':
        ttime, acc, fps = make_benchmark(net, path)
        print(f'[LOGS] Benchmark for {model} model')
        print(f'[LOGS] Total time {ttime}')
	print(f'[LOGS] Acc: {acc}')
        print(f'[LOGS] Fps: {fps}')


if __name__ == "__main__":
    main()
