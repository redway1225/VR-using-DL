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
### Training
```
usage: main_vdsr.py [-h] [--batchSize BATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--step STEP] [--cuda] [--resume RESUME]
               [--start-epoch START_EPOCH] [--clip CLIP] [--threads THREADS]
               [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY]
               [--pretrained PRETRAINED] [--gpus GPUS]
               
optional arguments:
  -h, --help            Show this help message and exit
  batchSize           Training batch size
  --nEpochs             Number of epochs to train for
  --lr                  Learning rate. Default=0.01
  --step                Learning rate decay, Default: n=10 epochs
  --cuda                Use cuda
  --resume              Path to checkpoint
  --clip                Clipping Gradients. Default=0.4
  --threads             Number of threads for data loader to use Default=1
  --momentum            Momentum, Default: 0.9
  --weight-decay        Weight decay, Default: 1e-4
  --pretrained PRETRAINED
                        path to pretrained model (default: none)
  --gpus GPUS           gpu ids (default: 0)
```
An example of training usage is shown as follows:
```
python main_vdsr.py --cuda --gpus 1 --resume checkpoint/model_epoch_50.pth --batchSize 64
```
## Results
<p>
  <img src='testing_lr_images/01.png' height='200' width='200'/>
  <img src='testing_lr_images_3x/01.png' height='200' width='200'/>
  <img src='result/01.png' height='200' width='200'/>
</p>
<p>
  <img src='testing_lr_images/06.png' height='192.77' width='200'/>
  <img src='testing_lr_images_3x/06.png' height='192.77' width='200'/>
  <img src='result/06.png' height='192.77' width='200'/>
</p>
<p>
  <img src='testing_lr_images/12.png' height='144.57' width='200'/>
  <img src='testing_lr_images_3x/12.png' height='144.57' width='200'/>
  <img src='result/12.png' height='144.57' width='200'/>
</p>

## Reference
* VDSR-Tensorflow
* https://github.com/DevKiHyun/VDSR-Tensorflow [1]
* PyTorch VDSR
* https://github.com/twtygqyy/pytorch-vdsr [ 2]

