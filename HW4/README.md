# HW4: Image-super-resolution
This homework is to train your image-super-resolution model to upscale the picture with an upscaling factor of 3.
## Hardware
* OS：Ubuntu 20.04 LTS
* CPU：Intel Xeon Gold 6136 @ 3.40 GHz
* GPU：NVIDIA GEFORCE RTX3090 + NVIDIA GEFORCE RTX2080
## Dataset
* Training set: 291 high-resolution images
* Testing set: 14 low-resolution images
## Requirements:
* python==3.6
* torch
* torchvision
* Pillow
* h5py
* numpy
## Quick start
* pip install -r requirements.txt
* git clone https://github.com/twtygqyy/pytorch-vdsr 
## Implementation
* Model：VDSR in CVPR’16
* Learning_rate：0.1->0.01
* Optimizer：SGD
* Batch_size ：64
* Epoches：120
## Code
* hw4_0856724.py：Train the model.
* pred_images.py：Test the model by inputting a path with testing dataset made by generate_test_mat.m, and out an pnsr result.
* demo_output.py：Generate high-resolution images.
* result：My best result which arrive psnr 22.507.
* data_gen_mat/generate_train.m：Make training set to .h5 files for training.
* data_gen_mat/generate_test_mat.m：Make testing set to 3 scales of .mat files for evaluation.
## Results
<p align="">
  <img width="510" height="510" src="https://github.com/redway1225/VR-using-DL/blob/master/HW4/result/01.png">
</p>
<p align="center">
  <img width="498" height="480" src="https://github.com/redway1225/VR-using-DL/blob/master/HW4/result/06.png">
</p>
<p align="center">
  <img width="498" height="360" src="https://github.com/redway1225/VR-using-DL/blob/master/HW4/result/12.png">
</p>

## Reference
* VDSR-Tensorflow
* https://github.com/DevKiHyun/VDSR-Tensorflow [1]
* PyTorch VDSR
* https://github.com/twtygqyy/pytorch-vdsr [ 2]

