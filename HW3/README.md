# HW3: Instance segmentation
This is to train your semantic segmentation to segment the objects with labels in testing pictures.
## Hardware
* OS：Ubuntu 20.04 LTS
* CPU：Intel Xeon Gold 6136 @ 3.40 GHz
* GPU：NVIDIA GEFORCE RTX3090 + NVIDIA GEFORCE RTX2080
## Dataset
* Tiny PASCAL VOC dataset, 1,449 images include many objects.
* 1,349 for training.
* 100 for testing.
## Class
* 20 common object classes
## Requirements:
* python==3.6
* torch 
* torchvision 
* cython
## Quick start
* pip install -r requirements.txt
* pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
* git clone https://github.com/facebookresearch/detectron2 detectron2
* pip install -e detectron2
## Implementation
* Ｍodel：detectron2 with ImageNet pre-trained model
* Learning_rate：5e-6 -> 0.00025
* Optimizer：SGD
* Batch_size per image：128
* Image per batch：2
* Eposide：30000
## Code
* hw3_0856724.py : Train the model.
* pred_images.py：Test the model by inputting an image, and out an image with detection result.
* make_json.py：Generate a json including testing dataset detection result.
* 0856724_11.json：My best json which arrive mAP 0.50511 with svhn dataset.
## Results
<p align="center">
  <img width="307" height="400" src="https://github.com/redway1225/VR-using-DL/blob/master/HW3/results/bike.png">
</p>
<p align="center">
  <img width="307" height="400" src="https://github.com/redway1225/VR-using-DL/blob/master/HW3/results/dogs.png">
</p>
<p align="center">
  <img width="307" height="400" src="https://github.com/redway1225/VR-using-DL/blob/master/HW3/results/boat.png">
</p>
## Reference
* Detectron2
* https://github.com/facebookresearch/detectron2 [1]
* Instance Segmentation using Detectron2
* https://github.com/AlessandroSaviolo/Instance-Segmentation-using-Detectron2 [ 2]
* Mask R-CNN for Object Detection and Segmentation
* https://github.com/matterport/Mask_RCNN [3]
* Image data preprocessing
* https://keras.io/api/preprocessing/image/ [4]
