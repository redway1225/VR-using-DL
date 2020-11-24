# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from natsort import natsorted
from yolo.utils.box import visualize_boxes
from yolo.config import ConfigParser

argparser = argparse.ArgumentParser(
    description='test yolov3 network with my digit detect weights')

argparser.add_argument(
    '-c',
    '--config',
    default="configs/svhn_test.json",
    help='config file')


argparser.add_argument(
    '-p',
    '--image_path',
    default="dataset/imgs/test",
    help='path to image file')


if __name__ == '__main__':
    args = argparser.parse_args()
    image_path   = args.image_path
    
    # 1. create yolo model & load weights
    config_parser = ConfigParser(args.config)
    model = config_parser.create_model(skip_detect_layer=False)
    detector = config_parser.create_detector(model)
    all_imgs = os.listdir(image_path)
    all_imgs = natsorted(all_imgs)
    all_results = []
    for i in range(len(all_imgs)):
        if (i % 100 == 0):
            print(i/100)
        one_image_path = os.path.join(image_path, all_imgs[i])
        
        # 2. Load image
        image = cv2.imread(one_image_path)
        image = image[:,:,::-1]
        
        # 3. Run detection
        boxes, labels, probs = detector.detect(image, 0.25)
        boxes_labels_probs = np.zeros((len(boxes), 6))
        
        for j in range(len(boxes)):
            boxes_labels_probs[j][0:4] = boxes[j]
            boxes_labels_probs[j][4] = labels[j]
            boxes_labels_probs[j][5] = probs[j]
        boxes_labels_probs = sorted(boxes_labels_probs, key=lambda x:x[5], reverse=True)
        
        for j in range(len(boxes)):
            boxes[j] = boxes_labels_probs[j][0:4]
            labels[j] = boxes_labels_probs[j][4]
            probs[j] = boxes_labels_probs[j][5]
        
            # make the result satisfy the requirment
            temp_for_swap = boxes[j][0]
            boxes[j][0] = boxes[j][1]
            boxes[j][1] = temp_for_swap
            temp_for_swap = boxes[j][2]
            boxes[j][2] = boxes[j][3]
            boxes[j][3] = temp_for_swap
            if labels[j] == 0:
                labels[j] = 10
        
        if(type(boxes) == list):
            this_img_result = {"bbox" : boxes, "score" : probs, "label" : labels}
        else:
            this_img_result = {"bbox" : boxes.tolist(), "score" : probs.tolist(), "label" : labels.tolist()}
        #print(this_img_result)
        all_results.append(this_img_result)
    jsObj = json.dumps(all_results) 
    emb_filename = ('0856724.json')   
    
    with open(emb_filename, "w") as f:  
        f.write(jsObj)  



