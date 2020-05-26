## Overview
**Neural network for object detection**

`$ git clone https://github.com/iamrishab/Safety-Equipments-Detection.git`

`$ cd Safety-Equipments-Detection`

`$ pip install -r requiremets.txt`

It is advisable that you do the installation in a virtual environment.

## Setup, Training and Inference of YOLO-v4 on Linux

Paper Yolo v4: https://arxiv.org/abs/2004.10934

* Yolo v4 Full comparison: [map_fps](https://user-images.githubusercontent.com/4096485/80283279-0e303e00-871f-11ea-814c-870967d77fd1.png)

## Results
#### Helmet
![Imgur](https://i.imgur.com/cvb2Zhd.jpg)

#### Mask
![Imgur](https://i.imgur.com/iPxTwyS.jpg)

## Requirements

* **CMake >= 3.12**: https://cmake.org/download/
* **CUDA 10.0**: https://developer.nvidia.com/cuda-toolkit-archive (on Linux do [Post-installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions))
* **OpenCV >= 2.4**: use your preferred package manager (brew, apt), build from source using [vcpkg](https://github.com/Microsoft/vcpkg) or download from [OpenCV official site](https://opencv.org/releases.html) 
* **cuDNN >= 7.0 for CUDA 10.0** https://developer.nvidia.com/rdp/cudnn-archive (on **Linux** copy `cudnn.h`,`libcudnn.so`... as desribed here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar
* **GCC or Clang**


## Setup

`$ git clone https://github.com/AlexeyAB/darknet.git`
`$ cd darknet`

### Compile on Linux (using `make`)

Before make,  set options in the `Makefile`:
* `GPU=1` to build with CUDA to accelerate by using GPU (CUDA should be in `/usr/local/cuda`)
* `CUDNN=1` to build with cuDNN v5-v7 to accelerate training by using GPU (cuDNN should be in `/usr/local/cudnn`)
* `OPENCV=1` to build with OpenCV 4.x/3.x/2.4.x - allows to detect on video files and video streams from network cameras or web-cams
* `OPENMP=1` to build with OpenMP support to accelerate Yolo by using multi-core CPU
* `LIBSO=1` to build a library `darknet.so` and binary runable file `uselib` that uses this library. 

Open a bash terminal inside the cloned repository and launch:

```bash
./build.sh
```

## Train to detect  custom objects

Model configuration and meta files are placed in `cfg` and `data` folder respectively. Download pretrained weights from [link](https://drive.google.com/file/d/1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp/view) and place it in darknet folder.

Step 1: Download the image and annotations.

Step 2: Change paths in `config.py` accordingly to save the parsed ground truths.

Step 3: Run `$ python parse_xml.py` to parse the ground truths from xml file to Yolo format.

Step 4: Copy paste the txt files from subsequent folder to image directory.

Step 5: Change path to files in `data/helmet.data` and `data/mask.data`. Place the model `cfg`,  `data`,  `names` files from `data` dir to `darknet/cfg` ,  `darknet/data`, `darknet/data` respectively.

Step 6: Start training using the command from the `darknet` folder. f.e.
`$ ./darknet detector train -dont_show data/helmet.data cfg/yolov4-helmet.cfg yolov4.conv.137`
 
 **Note:** If you changed width= or height= in your cfg-file, then new width and height must be divisible by 32.
  
  **Note:** if error `Out of memory` occurs then in `.cfg`-file you should increase `subdivisions=16`, 32 or 64
  
## Metrics for object detection
  
 **Evaluation metrics of the object detection problem**. 

### Intersection Over Union (IOU)

Intersection Over Union (IOU) is measure based on Jaccard Index that evaluates the overlap between two bounding boxes. It requires a ground truth bounding box ![](http://latex.codecogs.com/gif.latex?B_%7Bgt%7D) and a predicted bounding box ![](http://latex.codecogs.com/gif.latex?B_p). By applying the IOU we can tell if a detection is valid (True Positive) or not (False Positive).  
IOU is given by the overlapping area between the predicted bounding box and the ground truth bounding box divided by the area of union between them:  

<p align="center"> 
<img src="http://latex.codecogs.com/gif.latex?%5Ctext%7BIOU%7D%20%3D%20%5Cfrac%7B%5Ctext%7Barea%7D%20%5Cleft%28B_p%20%5Ccap%20B_%7Bgt%7D%5Cright%29%7D%7B%5Ctext%7Barea%7D%20%5Cleft%28B_p%20%5Ccup%20B_%7Bgt%7D%5Cright%29%7D">
</p>

### True Positive, False Positive, False Negative and True Negative  

Some basic concepts used by the metrics:  

* **True Positive (TP)**: A correct detection. Detection with IOU ≥ _threshold_  
* **False Positive (FP)**: A wrong detection. Detection with IOU < _threshold_  
* **False Negative (FN)**: A ground truth not detected  
* **True Negative (TN)**: Does not apply. It would represent a corrected misdetection. In the object detection task there are many possible bounding boxes that should not be detected within an image. Thus, TN would be all possible bounding boxes that were corrrectly not detected (so many possible boxes within an image). That's why it is not used by the metrics.

_threshold_: depending on the metric, it is usually set to 50%, 75% or 95%.

### Precision

Precision is the ability of a model to identify **only** the relevant objects. It is the percentage of correct positive predictions and is given by:

<p align="center"> 
<img src="http://latex.codecogs.com/gif.latex?Precision%20%3D%20%5Cfrac%7BTP%7D%7BTP&plus;FP%7D%3D%5Cfrac%7BTP%7D%7B%5Ctext%7Ball%20detections%7D%7D">
</p>

### Recall 

Recall is the ability of a model to find all the relevant cases (all ground truth bounding boxes). It is the percentage of true positive detected among all relevant ground truths and is given by:

<p align="center"> 
<img src="http://latex.codecogs.com/gif.latex?Recall%20%3D%20%5Cfrac%7BTP%7D%7BTP&plus;FN%7D%3D%5Cfrac%7BTP%7D%7B%5Ctext%7Ball%20ground%20truths%7D%7D">
</p>

### Precision x Recall curve

The Precision x Recall curve is a good way to evaluate the performance of an object detector as the confidence is changed by plotting a curve for each object class. An object detector of a particular class is considered good if its precision stays high as recall increases, which means that if you vary the confidence threshold, the precision and recall will still be high. Another way to identify a good object detector is to look for a detector that can identify only relevant objects (0 False Positives = high precision), finding all ground truth objects (0 False Negatives = high recall).  

### Average Precision

AP is the precision averaged across all recall values between 0 and 1.  

## How to evaluation
Copy trained weights for **mask detection** from [link](https://drive.google.com/file/d/1OQuJBY7d2HRs6PYmYmxhyRiZhrx3VEjU/view?usp=sharing).  Please note the model is trained for only 4000 epochs due to lack of resources, it takes atleast 10000 epochs for the model to converge.

Copy trained weights for **helmet detection** from [link](https://drive.google.com/file/d/1Yzo_Sq6xIyLh3AKGCvPf_NawjWuOUKJO/view?usp=sharing).  Please note the model is trained for only 1000 epochs due to lack of resources, it takes atleast 10000 epochs for the model to converge.


Change path in `config.py` accordingly. Follow the steps below to start evaluating model detections:

1. Create the ground truth files: `$ python generate_eval_gt.py`
2. Create your detection files: `$ python generate_eval_pred.py`
3. Go to Object-Detection-Metrics folder: `$ cd Object-Detection-Metrics folder` 
4. Run the command for mask detection evaluation: `python pascalvoc.py -gt path/to/gt-mask/dir -det path/to/pred-mask/dir -gtformat xyrb -detformat xyrb`  
5. Run the command for helmet detection evaluation: `python pascalvoc.py -gt path/to/gt-helmet/dir -det path/to/pred-helmet/dir -gtformat xyrb -detformat xyrb` 
