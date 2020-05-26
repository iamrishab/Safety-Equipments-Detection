## Overview
Neural network for object detection

## Setup, Training and Inference of YOLO-v4 on Linux

Paper Yolo v4: https://arxiv.org/abs/2004.10934

**GeForce RTX 2080 Ti:**
| Network Size 	| Darknet, FPS (avg)| tkDNN TensorRT FP32, FPS  | tkDNN TensorRT FP16, FPS  | OpenCV FP16, FPS | tkDNN TensorRT FP16 batch=4, FPS  | OpenCV FP16 batch=4, FPS | tkDNN Speedup |
|:-----:|:--------:|--------:|--------:|--------:|--------:|--------:|------:|
|320	| 100 | 116 | **202** | 171 | **423** | 384 | **4.2x** |
|416	| 82 | 103 | **162** | 146 | **284** | 260 | **3.5x** |
|512	| 69 | 91 | **134** | 125 | **206** | 190 | **2.9x** |
|608 	| 53 | 62 | **103** | 100 | **150** | 133 | **2.8x**  |

* Yolo v4 Full comparison: [map_fps](https://user-images.githubusercontent.com/4096485/80283279-0e303e00-871f-11ea-814c-870967d77fd1.png)

## Results
#### Helmet
![Imgur](https://i.imgur.com/cvb2Zhd.jpg)

#### Mask
![Imgur](https://i.imgur.com/iPxTwyS.jpg)

#### How to evaluate AP of YOLOv4


#### Pre-trained models

There are weights-file for different cfg-files (trained for MS COCO dataset):

FPS on RTX 2070 (R) and Tesla V100 (V):

* [yolov4.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg) - 245 MB: [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) (Google-drive mirror [yolov4.weights](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT) ) paper [Yolo v4](https://arxiv.org/abs/2004.10934)
    just change `width=` and `height=` parameters in `yolov4.cfg` file and use the same `yolov4.weights` file for all cases:
    * `width=608 height=608` in cfg: **65.7% mAP@0.5 (43.5% AP@0.5:0.95) - 34(R) FPS / 62(V) FPS** - 128.5 BFlops
    * `width=512 height=512` in cfg: **64.9% mAP@0.5 (43.0% AP@0.5:0.95) - 45(R) FPS / 83(V) FPS** - 91.1 BFlops
    * `width=416 height=416` in cfg: **62.8% mAP@0.5 (41.2% AP@0.5:0.95) - 55(R) FPS / 96(V) FPS** - 60.1 BFlops
    * `width=320 height=320` in cfg:   **60% mAP@0.5 (  38% AP@0.5:0.95) - 63(R) FPS / 123(V) FPS** - 35.5 BFlops

You can get cfg-files by path: `darknet/cfg/`

### Requirements

* **CMake >= 3.12**: https://cmake.org/download/
* **CUDA 10.0**: https://developer.nvidia.com/cuda-toolkit-archive (on Linux do [Post-installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions))
* **OpenCV >= 2.4**: use your preferred package manager (brew, apt), build from source using [vcpkg](https://github.com/Microsoft/vcpkg) or download from [OpenCV official site](https://opencv.org/releases.html) (on Windows set system variable `OpenCV_DIR` = `C:\opencv\build` - where are the `include` and `x64` folders [image](https://user-images.githubusercontent.com/4096485/53249516-5130f480-36c9-11e9-8238-a6e82e48c6f2.png))
* **cuDNN >= 7.0 for CUDA 10.0** https://developer.nvidia.com/rdp/cudnn-archive (on **Linux** copy `cudnn.h`,`libcudnn.so`... as desribed here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar , on **Windows** copy `cudnn.h`,`cudnn64_7.dll`, `cudnn64_7.lib` as desribed here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installwindows )
* **GPU with CC >= 3.0**: https://en.wikipedia.org/wiki/CUDA#GPUs_supported
* on Linux **GCC or Clang**

#### How to use on the command line

Find executable file `./darknet` in the root directory

* Yolo v4 COCO - **image**: `./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -thresh 0.25`
* **Output coordinates** of objects: `./darknet detector test cfg/coco.data yolov4.cfg yolov4.weights -ext_output dog.jpg`
* Yolo v4 COCO - **video**: `./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights -ext_output test.mp4`
* Yolo v4 COCO - **WebCam 0**: `./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights -c 0`
* **To check accuracy mAP@IoU=50**: `./darknet detector map data/obj.data yolo-obj.cfg backup\yolo-obj_7000.weights`
* ** To check accuracy mAP@IoU=75**:  `./darknet detector map data/obj.data yolo-obj.cfg backup\yolo-obj_7000.weights -iou_thresh 0.75`

### How to compile on Linux (using `make`)

Just do `make` in the darknet directory. (You can try to compile and run it on Google Colab in cloud [link](https://colab.research.google.com/drive/12QusaaRj_lUwCGDvQNfICpa7kA7_a2dE) (press «Open in Playground» button at the top-left corner) and watch the video [link](https://www.youtube.com/watch?v=mKAEGSxwOAY) )
Before make, you can set such options in the `Makefile`: [link](https://github.com/AlexeyAB/darknet/blob/9c1b9a2cf6363546c152251be578a21f3c3caec6/Makefile#L1)

* `GPU=1` to build with CUDA to accelerate by using GPU (CUDA should be in `/usr/local/cuda`)
* `CUDNN=1` to build with cuDNN v5-v7 to accelerate training by using GPU (cuDNN should be in `/usr/local/cudnn`)
* `CUDNN_HALF=1` to build for Tensor Cores (on Titan V / Tesla V100 / DGX-2 and later) speedup Detection 3x, Training 2x
* `OPENCV=1` to build with OpenCV 4.x/3.x/2.4.x - allows to detect on video files and video streams from network cameras or web-cams
* `DEBUG=1` to bould debug version of Yolo
* `OPENMP=1` to build with OpenMP support to accelerate Yolo by using multi-core CPU
* `LIBSO=1` to build a library `darknet.so` and binary runable file `uselib` that uses this library. 

Open a bash terminal inside the cloned repository and launch:

```bash
./build.sh
```
To run Darknet on Linux use examples from this article, just use `./darknet` instead of `darknet.exe`, i.e. use this command: `./darknet detector test ./cfg/coco.data ./cfg/yolov4.cfg ./yolov4.weights`

## How to train with multi-GPU:

1. Train it first on 1 GPU for like 1000 iterations: `darknet.exe detector train cfg/coco.data cfg/yolov4.cfg yolov4.conv.137`

2. Then stop and by using partially-trained model `/backup/yolov4_1000.weights` run training with multigpu (up to 4 GPUs): `darknet.exe detector train cfg/coco.data cfg/yolov4.cfg /backup/yolov4_1000.weights -gpus 0,1,2,3`

If you get a Nan, then for some datasets better to decrease learning rate, for 4 GPUs set `learning_rate = 0,00065` (i.e. learning_rate = 0.00261 / GPUs). In this case also increase 4x times `burn_in =` in your cfg-file. I.e. use `burn_in = 4000` instead of `1000`.

https://groups.google.com/d/msg/darknet/NbJqonJBTSY/Te5PfIpuCAAJ

## How to train (to detect your custom objects):

Training Yolo v4:

0. For training `cfg/yolov4-custom.cfg` download the pre-trained weights-file (162 MB): [yolov4.conv.137](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137) (Google drive mirror [yolov4.conv.137](https://drive.google.com/open?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp) )

1. Create file `yolo-obj.cfg` with the same content as in `yolov4-custom.cfg` (or copy `yolov4-custom.cfg` to `yolo-obj.cfg)` and:

  * change line batch to [`batch=64`]
  * change line subdivisions to [`subdivisions=16`]  
  * change line max_batches to (`classes*2000` but not less than number of training images, and not less than `6000`), f.e. [`max_batches=6000`] if you train for 3 classes.
  * change line steps to 80% and 90% of max_batches, f.e. [`steps=4800,5400`]
  * set network size `width=416 height=416` or any value multiple of 32
  * change line `classes=80` to your number of objects in each of 3 `[yolo]`-layers
  * change [`filters=255`] to filters=(classes + 5)x3 in the 3 `[convolutional]` before each `[yolo]` layer, keep in mind that it only has to be the last `[convolutional]` before each of the `[yolo]` layers. So if `classes=1` then should be `filters=18`. If `classes=2` then write `filters=21`.
  
  **(Do not write in the cfg-file: filters=(classes + 5)x3)**
  
  (Generally `filters` depends on the `classes`, `coords` and number of `mask`s, i.e. filters=`(classes + coords + 1)*<number of mask>`, where `mask` is indices of anchors. If `mask` is absence, then filters=`(classes + coords + 1)*num`)

  So for example, for 2 objects, your file `yolo-obj.cfg` should differ from `yolov4-custom.cfg` in such lines in each of **3** [yolo]-layers:

  ```
  [convolutional]
  filters=21

  [region]
  classes=2
  ```

2. Create file `obj.names` in the directory `data/`, with objects names - each in new line

3. Create file `obj.data` in the directory `data/`, containing (where **classes = number of objects**):

  ```
  classes= 2
  train  = data/train.txt
  valid  = data/test.txt
  names = data/obj.names
  backup = backup/
  ```

5. You should label each object on images from your dataset. 
Put `.txt`-file for each `.jpg`-image-file - in the same directory and with the same name, but with `.txt`-extension, and put to file: object number and object coordinates on this image, for each object in new line: 

`<object-class> <x_center> <y_center> <width> <height>`

  Where: 
  * `<object-class>` - integer object number from `0` to `(classes-1)`
  * `<x_center> <y_center> <width> <height>` - float values **relative** to width and height of image, it can be equal from `(0.0 to 1.0]`
  * for example: `<x> = <absolute_x> / <image_width>` or `<height> = <absolute_height> / <image_height>`
  * atention: `<x_center> <y_center>` - are center of rectangle (are not top-left corner)

  For example for `img1.jpg` you will be created `img1.txt` containing:

  ```
  1 0.716797 0.395833 0.216406 0.147222
  0 0.687109 0.379167 0.255469 0.158333
  1 0.420312 0.395833 0.140625 0.166667
  ```

6. Create file `train.txt` with filenames of your images, each filename in new line, with path relative to `darknet`, for example containing:

  ```
  data/obj/img1.jpg
  data/obj/img2.jpg
  data/obj/img3.jpg
  ```

7. Download pre-trained weights for the convolutional layers and put to the directory `build\darknet\x64`
    * for `yolov4.cfg`, `yolov4-custom.cfg` (162 MB): [yolov4.conv.137](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137) (Google drive mirror [yolov4.conv.137](https://drive.google.com/open?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp) )

8. Start training by using the command line: `darknet.exe detector train data/obj.data yolo-obj.cfg yolov4.conv.137`
     
   To train on Linux use command: `./darknet detector train data/obj.data yolo-obj.cfg yolov4.conv.137` (just use `./darknet` instead of `darknet.exe`)
     
   * (file `yolo-obj_last.weights` will be saved to the `build\darknet\x64\backup\` for each 100 iterations)
   * (file `yolo-obj_xxxx.weights` will be saved to the `build\darknet\x64\backup\` for each 1000 iterations)
   * (to disable Loss-Window use `./darknet detector train data/obj.data yolo-obj.cfg yolov4.conv.137 -dont_show`, if you train on computer without monitor like a cloud Amazon EC2)
   * (to see the mAP & Loss-chart during training on remote server without GUI, use command `./darknet detector train data/obj.data yolo-obj.cfg yolov4.conv.137 -dont_show -mjpeg_port 8090 -map` then open URL `http://ip-address:8090` in Chrome/Firefox browser)

8.1. For training with mAP (mean average precisions) calculation for each 4 Epochs (set `valid=valid.txt` or `train.txt` in `obj.data` file) and run: `darknet.exe detector train data/obj.data yolo-obj.cfg yolov4.conv.137 -map`

9. After training is complete - get result `yolo-obj_final.weights` from path `build\darknet\x64\backup\`

 * After each 100 iterations you can stop and later start training from this point. For example, after 2000 iterations you can stop training, and later just start training using: `darknet.exe detector train data/obj.data yolo-obj.cfg backup\yolo-obj_2000.weights`

    (in the original repository https://github.com/pjreddie/darknet the weights-file is saved only once every 10 000 iterations `if(iterations > 1000)`)

 * Also you can get result earlier than all 45000 iterations.
 
 **Note:** If during training you see `nan` values for `avg` (loss) field - then training goes wrong, but if `nan` is in some other lines - then training goes well.
 
 **Note:** If you changed width= or height= in your cfg-file, then new width and height must be divisible by 32.
 
 **Note:** After training use such command for detection: `./darknet detector test data/obj.data yolo-obj.cfg yolo-obj_8000.weights`
 
  **Note:** if error `Out of memory` occurs then in `.cfg`-file you should increase `subdivisions=16`, 32 or 64: [link](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L4)

# Metrics for object detection
  
 **evaluation metrics of the object detection problem**. 


## Important definitions  

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

## Metrics

In the topics below there are some comments on the most popular metrics used for object detection.

### Precision x Recall curve

The Precision x Recall curve is a good way to evaluate the performance of an object detector as the confidence is changed by plotting a curve for each object class. An object detector of a particular class is considered good if its precision stays high as recall increases, which means that if you vary the confidence threshold, the precision and recall will still be high. Another way to identify a good object detector is to look for a detector that can identify only relevant objects (0 False Positives = high precision), finding all ground truth objects (0 False Negatives = high recall).  

A poor object detector needs to increase the number of detected objects (increasing False Positives = lower precision) in order to retrieve all ground truth objects (high recall). That's why the Precision x Recall curve usually starts with high precision values, decreasing as recall increases. You can see an example of the Prevision x Recall curve in the next topic (Average Precision). This kind of curve is used by the PASCAL VOC 2012 challenge and is available in our implementation.  

### Average Precision

Another way to compare the performance of object detectors is to calculate the area under the curve (AUC) of the Precision x Recall curve. As AP curves are often zigzag curves going up and down, comparing different curves (different detectors) in the same plot usually is not an easy task - because the curves tend to cross each other much frequently. That's why Average Precision (AP), a numerical metric, can also help us compare different detectors. In practice AP is the precision averaged across all recall values between 0 and 1.  

From 2010 on, the method of computing AP by the PASCAL VOC challenge has changed. Currently, **the interpolation performed by PASCAL VOC challenge uses all data points, rather than interpolating only 11 equally spaced points as stated in their [paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.5766&rep=rep1&type=pdf)**. As we want to reproduce their default implementation, our default code (as seen further) follows their most recent application (interpolating all data points). However, we also offer the 11-point interpolation approach. 

## How to evaluation

Follow the steps below to start evaluating your detections:

1. [Create the ground truth files](#create-the-ground-truth-files)
2. [Create your detection files](#create-your-detection-files)
3. For **Pascal VOC metrics**, run the command: `python pascalvoc.py`  
   If you want to reproduce the example above, run the command: `python pascalvoc.py -t 0.3`
4. (Optional) [You can use arguments to control the IOU threshold, bounding boxes format, etc.](#optional-arguments)

### Create the ground truth files

- Create a separate ground truth text file for each image in the folder **groundtruths/**.
- In these files each line should be in the format: `<class_name> <left> <top> <right> <bottom>`.    
- E.g. The ground truth bounding boxes of the image "2008_000034.jpg" are represented in the file "2008_000034.txt":
  ```
  bottle 6 234 45 362
  person 1 156 103 336
  person 36 111 198 416
  person 91 42 338 500
  ```
    
If you prefer, you can also have your bounding boxes in the format: `<class_name> <left> <top> <width> <height>` (see here [**\***](#asterisk) how to use it). In this case, your "2008_000034.txt" would be represented as:
  ```
  bottle 6 234 39 128
  person 1 156 102 180
  person 36 111 162 305
  person 91 42 247 458
  ```

### Create your detection files

- Create a separate detection text file for each image in the folder **detections/**.
- The names of the detection files must match their correspond ground truth (e.g. "detections/2008_000182.txt" represents the detections of the ground truth: "groundtruths/2008_000182.txt").
- In these files each line should be in the following format: `<class_name> <confidence> <left> <top> <right> <bottom>` (see here [**\***](#asterisk) how to use it).
- E.g. "2008_000034.txt":
    ```
    bottle 0.14981 80 1 295 500  
    bus 0.12601 36 13 404 316  
    horse 0.12526 430 117 500 307  
    pottedplant 0.14585 212 78 292 118  
    tvmonitor 0.070565 388 89 500 196  
    ```

Also if you prefer, you could have your bounding boxes in the format: `<class_name> <left> <top> <width> <height>`.

### Optional arguments

Optional arguments:

| Argument &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Description | Example | Default |
|:-------------:|:-----------:|:-----------:|:-----------:|
| `-h`,<br>`--help ` |	show help message | `python pascalvoc.py -h` | |  
|  `-v`,<br>`--version` | check version | `python pascalvoc.py -v` | |  
| `-gt`,<br>`--gtfolder` | folder that contains the ground truth bounding boxes files | `python pascalvoc.py -gt /home/whatever/my_groundtruths/` | `/Object-Detection-Metrics/groundtruths`|  
| `-det`,<br>`--detfolder` | folder that contains your detected bounding boxes files | `python pascalvoc.py -det /home/whatever/my_detections/` | `/Object-Detection-Metrics/detections/`|  
| `-t`,<br>`--threshold` | IOU thershold that tells if a detection is TP or FP | `python pascalvoc.py -t 0.75` | `0.50` |  
| `-gtformat` | format of the coordinates of the ground truth bounding boxes [**\***](#asterisk) | `python pascalvoc.py -gtformat xyrb` | `xywh` |
| `-detformat` | format of the coordinates of the detected bounding boxes [**\***](#asterisk) | `python pascalvoc.py -detformat xyrb` | `xywh` | |  
| `-gtcoords` | reference of the ground truth bounding bounding box coordinates.<br>If the annotated coordinates are relative to the image size (as used in YOLO), set it to `rel`.<br>If the coordinates are absolute values, not depending to the image size, set it to `abs` |  `python pascalvoc.py -gtcoords rel` | `abs` |  
| `-detcoords` | reference of the detected bounding bounding box coordinates.<br>If the coordinates are relative to the image size (as used in YOLO), set it to `rel`.<br>If the coordinates are absolute values, not depending to the image size, set it to `abs` | `python pascalvoc.py -detcoords rel` | `abs` |  
| `-imgsize ` | image size in the format `width,height` <int,int>.<br>Required if `-gtcoords` or `-detcoords` is set to `rel` | `python pascalvoc.py -imgsize 600,400` |  
| `-sp`,<br>`--savepath` | folder where the plots are saved | `python pascalvoc.py -sp /home/whatever/my_results/` | `Object-Detection-Metrics/results/` |  
| `-np`,<br>`--noplot` | if present no plot is shown during execution | `python pascalvoc.py -np` | not presented.<br>Therefore, plots are shown |  

<a name="asterisk"> </a>
(**\***) set `-gtformat xywh` and/or `-detformat xywh` if format is `<left> <top> <width> <height>`. Set to `-gtformat xyrb` and/or `-detformat xyrb`  if format is `<left> <top> <right> <bottom>`.
  