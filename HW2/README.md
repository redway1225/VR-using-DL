# HW2: Object detection
This is to train your detector to detect the number in street view pictures.
## Hardware
* OS：Ubuntu 20.04 LTS
* CPU：Intel Xeon Gold 6136 @ 3.40 GHz
* GPU：NVIDIA GEFORCE RTX3090 + NVIDIA GEFORCE RTX2080
## Dataset
* 46,470 images include many digits 
* 33,402 for training
* 13,068 for testing
## Class
10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9
and '0' has label 10. .
## Requirements:
* python==3.6
* tensorflow-GPU==2.0.0-beta1
* opencv-python
* Pillow
* requests
* tqdm
* sklearn
* imgaug==0.2.6
* pytest-cov
* codecov
* matplotlib
* natsort
## Quick start
* First, download yolov3 from reference [1], and follow its steps to setup the yolov3.
* Second, download the SVHN VOC Annotations from reference [2] to train.
## Implementation
* Ｍodel：yolov3 with yolov3.weights(pretrained)
* Loss function：binary cross-entropy loss
* Learning_rate：5e-6 (Decrease by 10% when the model stops
growing.)
* Batch_size：16（32 will out of GPU memory)
## Code
* HW2_0856724.py : Train the model.
* pred.py：Test the model by inputting an image, and out an image with detection result.
* HW2_0856724.ipynb：Test the speed of model.
* gen_json.py：Generate a json including testing dataset detection result.
* 0856724_5.json：My best json which arrive mAP 0.44333 with svhn dataset.
## Reference
TF2 eager implementation of Yolo-v3
https://github.com/penny4860/tf2-eager-yolo3 [1]
SVHN Voc Annotation Format
https://github.com/penny4860/svhn-voc-annotation-format [2]
