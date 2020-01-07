### Requirements
```
conda create -n face python==3.7 tensorflow-gpu==1.14
conda install -c conda-forge dlib
conda activate face

pip install \
  click \
  tqdm \
  mtcnn
```

Build opencv with cuda backend by [this](https://github.com/bartoszptak/Efficient_Object_Detection_Algorithms_Research/blob/master/INSTALLATION_GUIDE.md) instruction.

### Installation

```
git clone git@github.com:bartoszptak/Face_detection_algorithms_research.git
cd Face_detection_algorithms_research/
```

```
download https://docs.google.com/uc?export=download&id=13gFDLFhhBqwMw6gf8jVUvNDH2UrgCCrX
cp ~/Downloads/yolov3-wider_16000.weights.zip models/yoloface/yolov3-wider_16000.weights.zip
unzip models/yoloface/yolov3-wider_16000.weights.zip -d models/yoloface/
```

```
cd data/
git clone git@github.com:penolove/FDDB_DataSet_4_faster_rcnn.git
cd FDDB_DataSet_4_faster_rcnn/
sh get_data.sh
sh generate_FDDB_2010.sh
```

```
cd ../../
git clone git@github.com:Cartucho/mAP.git
```
