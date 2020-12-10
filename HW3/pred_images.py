import torch, torchvision
import numpy as np
import pandas as pd
import random
import json
import cv2
import os

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
setup_logger()

import matplotlib.pyplot as plt
from pycocotools.coco import COCO




cfg = get_cfg()
cfg.merge_from_file("gdrive/MyDrive/HW3/configs/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

cfg.MODEL.WEIGHTS = 'gdrive/MyDrive/HW3/model/model_0029999.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20

register_coco_instances("V", {}, "gdrive/My Drive/HW3/dataset/test/test.json", "gdrive/My Drive/HW3/dataset/test/test_images")
test_metadata = MetadataCatalog.get("V")
test_dataset_dicts = DatasetCatalog.get("V")

cfg.DATASETS.TEST = ("V")
predictor = DefaultPredictor(cfg)
coco = COCO("gdrive/My Drive/HW3/dataset/train/pascal_train.json") 

from detectron2.utils.visualizer import ColorMode

for d in random.sample(test_dataset_dicts, 3):    
    image = cv2.imread(d["file_name"])
    outputs = predictor(image)
    v = Visualizer(image[:, :, ::-1],
                   metadata=test_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW  
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(v.get_image()[:, :, ::-1])