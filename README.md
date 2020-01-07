# Face detection algorithms research

### Algorithms
* [Frontal face cascade classifier from OpenCV](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)
* [Frontal face detector from dlib](http://dlib.net/face_detector.py.html)
* [CNN face detection model from dlib](http://dlib.net/cnn_face_detector.py.html)
* [yoloface model](https://github.com/sthanhng/yoloface)
* [face MTCNN model](https://github.com/ipazc/mtcnn)
* [DNN Face Detector](https://github.com/spmallick/learnopencv/tree/master/AgeGender)

### Dataset
**FDDB_DataSet** https://github.com/penolove/FDDB_DataSet_4_faster_rcnn  
,,This data set contains the annotations for 5171 faces in a set of 2850 images taken from the Faces in the Wild data set.''

### Metrics
**mAP** https://github.com/Cartucho/mAP  
,,The performance of your neural net will be judged using the mAP criterium defined in the PASCAL VOC 2012 competition.''

### Results
*Total time - total process time with loading time of libraries and model  
**FPS - images per second, only the total time of the interference divided by the number of examples

* benchmarking on Intel Core i5-8400

| Model<br>name | *Total<br>time (s) | **FPS<br>(img/s) | FDDB 2010<br>mAP@0.5 |
|:-----------------------------------------------:|:------------------:|:----------------:|:--------------------:|
| Frontal face cascade <br>classifier from OpenCV | 83.57 | 32.90 | 0.0001 |
| Frontal face detector<br>from dlib | 135.31 | 20.33 | 0.5227 |
| CNN face detection <br>model from dlib |  |  |  |
| yoloface model |  |  |  |
| face MTCNN model | 186.82 | 14.72 | 0.7745 |
| DNN Face Detector | 34.50 | 79.71 | 0.7008 |

* benchmarking on Intel Core i5-8400 with NVIDIA GTX1060

| Model<br>name | *Total<br>time (s) | **FPS<br>(img/s) | FDDB 2010<br>mAP@0.5 |
|:-----------------------------------------------:|:------------------:|:----------------:|:--------------------:|
| Frontal face cascade <br>classifier from OpenCV | - | - | 0.0001 |
| Frontal face detector<br>from dlib | 135.32 | 20.32 | 0.5227 |
| CNN face detection <br>model from dlib |  |  |  |
| yoloface model |  |  |  |
| face MTCNN model | 122.74 | 22.41 | 0.7745 |
| DNN Face Detector | 16.25 | 169.12 | 0.7008 |

### Conclusions
**I leave the subject open for you. I only share the results, with a friendly model-prediction interface. You should decide for yourself whether you need FPS or high mAP.**
