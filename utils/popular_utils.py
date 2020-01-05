import cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET
import os
import time
import subprocess


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

def make_benchmark(net, 
                   data_path=os.path.join('data', 'FDDB_DataSet_4_faster_rcnn', 'FDDB_2010'),
                   gt_path = os.path.join('mAP','input','ground-truth'),
                   pred_path =os.path.join('mAP','input','detection-results')
                   ):

    xmls = glob.glob(os.path.join(data_path, 'Annotations/*.xml'))
    print(f'[LOGS] Found {len(xmls)} annotations')    

    for pt in glob.glob(os.path.join(gt_path, '*.txt'))+glob.glob(os.path.join(pred_path, '*.txt')):
        os.remove(pt)

    tim = 0.0
    imgs = 0

    for xml_file in xmls:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        filename = root.find('filename').text
        img = cv2.imread(os.path.join(data_path, f'JPEGImages/{filename}.jpg'))
        
        start = time.time()
        preds = net.predict(img)
        stop = time.time()
        
        tim += (stop-start)
        imgs +=1
        
        with open(os.path.join(pred_path, f'{filename}.txt'), 'w') as f:
            for xmin, ymin, xmax, ymax in preds:
                print(f'face 1.0 {xmin} {ymin} {xmax} {ymax}', file=f)
            
        
        with open(os.path.join(gt_path, f'{filename}.txt'), 'w') as f:
            for ob in root.findall('object'):
                bndbox = ob.find('bndbox')
                xmin, ymin, xmax, ymax = bndbox.find("xmin").text, bndbox.find("ymin").text, bndbox.find("xmax").text, bndbox.find("ymax").text
                print(f'face {xmin} {ymin} {xmax} {ymax}', file=f)

    os.remove(os.path.join('mAP', 'results', 'results.txt'))

    print('[LOGS] Calculate mAP')
    process = subprocess.Popen(['python', 'mAP/main.py', '--no-animation', '--no-plot', '--quiet'], stdout=subprocess.PIPE)
    stdout = process.communicate()[0]

    return tim, float(stdout.decode().split()[-1][:-1])/100, imgs/tim
