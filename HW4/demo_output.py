import argparse, os
import torch
from torch.autograd import Variable
from scipy.ndimage import imread
from scipy.misc import imsave
from PIL import Image
import numpy as np
import time, math
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="PyTorch VDSR Demo")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_epoch_50.pth", type=str, help="model path")
parser.add_argument("--image", default="testing_lr_images", type=str, help="image name")
parser.add_argument("--scale", default=3, type=int, help="scale factor, Default: 3")
parser.add_argument("--gpus", default="1", type=str, help="gpu ids (default: 1)")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)
    
def colorize(y, ycbcr): 
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycbcr[:,:,1]
    img[:,:,2] = ycbcr[:,:,2]
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img

opt = parser.parse_args()
cuda = opt.cuda
image_dir = opt.image
image_names = os.listdir(image_dir)
scale = opt.scale
if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")


model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]
for img_name in image_names:
    img = Image.open(image_dir + "/" + img_name)
    (w, h) = img.size
    new_img = img.resize((3*w, scale*h),Image.ANTIALIAS)
    new_img.save(image_dir + "_"+str(scale)+"x/" + img_name)
    
    im_b_ycbcr = imread(image_dir + "_"+str(scale)+"x/" + img_name, mode="YCbCr")
    
    im_b_y = im_b_ycbcr[:,:,0].astype(float)
    
    
    
    im_input = im_b_y/255.
    
    im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])
    
    if cuda:
        model = model.cuda()
        im_input = im_input.cuda()
    else:
        model = model.cpu()
    
    start_time = time.time()
    out = model(im_input)
    elapsed_time = time.time() - start_time
    
    out = out.cpu()
    
    im_h_y = out.data[0].numpy().astype(np.float32)
    
    im_h_y = im_h_y * 255.
    im_h_y[im_h_y < 0] = 0
    im_h_y[im_h_y > 255.] = 255.
    
    
    
    im_h = colorize(im_h_y[0,:,:], im_b_ycbcr)
    
    im_b = Image.fromarray(im_b_ycbcr, "YCbCr").convert("RGB")
    
    print("Scale=",opt.scale)
    
    print("It takes {}s for processing".format(elapsed_time))
    
    imsave("result/"  + img_name, im_h)
"""
ax = plt.subplot("132")
ax.imshow(im_b)
ax.set_title("Input(bicubic)")

ax = plt.subplot("133")
ax.imshow(im_h)
ax.set_title("Output(vdsr)")
plt.show()
imsave('00_4.png', im_h)
"""