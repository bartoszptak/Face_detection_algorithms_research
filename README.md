# Face detection research

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
| Frontal face cascade <br>classifier from OpenCV |  |  |  |
| Frontal face detector<br>from dlib |  |  |  |
| CNN face detection <br>model from dlib |  |  |  |
| yoloface model |  |  |  |
| face MTCNN model |  |  | |
| DNN Face Detector |  |  |  |

* benchmarking on Intel Core i5-8400 with NVIDIA GTX1060

| Model<br>name | *Total<br>time (s) | **FPS<br>(img/s) | FDDB 2010<br>mAP@0.5 |
|:-----------------------------------------------:|:------------------:|:----------------:|:--------------------:|
| Frontal face cascade <br>classifier from OpenCV |  |  |  |
| Frontal face detector<br>from dlib |  |  |  |
| CNN face detection <br>model from dlib |  |  |  |
| yoloface model |  |  |  |
| face MTCNN model | | |  |
| DNN Face Detector |  |  |  |

### Conclusions
**I leave the subject open for you. I only share the results, with a friendly model-prediction interface. You should decide for yourself whether you need FPS or high mAP.**
