import torch, torchvision
import numpy as np
import pandas as pd
import random
import json
import cv2
import os

import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
setup_logger()

import itertools
from itertools import groupby
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskutil
from pycocotools import mask as maskUtils

register_coco_instances("T", {}, "gdrive/My Drive/HW3/dataset/train/pascal_train.json", "gdrive/My Drive/HW3/dataset/train/train_images")


# metadata
train_metadata = MetadataCatalog.get("T")
test_metadata = MetadataCatalog.get("V")

# dataset dictionary
train_dataset_dicts = DatasetCatalog.get("T")
test_dataset_dicts = DatasetCatalog.get("V")

cfg = get_cfg()
cfg.merge_from_file("gdrive/MyDrive/HW3/configs/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

# load dataset
cfg.DATASETS.TRAIN = ("T")
cfg.DATASETS.TEST = ()
# parameters
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 20000                       
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
cfg.MODEL.WEIGHTS = "gdrive/MyDrive/HW3/model/model_0029999.pth"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

