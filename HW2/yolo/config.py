import json
import os
import glob
from yolo.net import Yolonet
from yolo.dataset.generator import BatchGenerator
from yolo.utils.utils import download_if_not_exists
from yolo.frontend import YoloDetector
from yolo.evaluate import Evaluator

class ConfigParser(object):
    def __init__(self, config_file):
        with open(config_file) as data_file:    
            config = json.load(data_file)
        
        self._model_config = config["model"]
        self._pretrained_config = config["pretrained"]
        self._train_config = config["train"]
        
    def create_model(self, skip_detect_layer=True):
        model = Yolonet(n_classes=len(self._model_config["labels"]))
        
        keras_weights = self._pretrained_config["keras_format"]
        if os.path.exists(keras_weights):
            model.load_weights(keras_weights)
            print("Keras pretrained weights loaded from {}!!".format(keras_weights))
        else:
            download_if_not_exists(self._pretrained_config["darknet_format"],
                                   "https://pjreddie.com/media/files/yolov3.weights")

            model.load_darknet_params(self._pretrained_config["darknet_format"], skip_detect_layer)
            print("Original yolov3 weights loaded!!")

        return model

    def create_detector(self, model):
        d = YoloDetector(model, self._model_config["anchors"], net_size=self._model_config["net_size"])
        return d

    def create_generator(self):
        train_ann_fnames = self._get_train_anns()
        valid_ann_fnames = self._get_valid_anns()
    
        train_generator = BatchGenerator(train_ann_fnames,
                                         self._train_config["train_image_folder"],
                                         batch_size=self._train_config["batch_size"],
                                         labels=self._model_config["labels"],
                                         anchors=self._model_config["anchors"],
                                         min_net_size=self._train_config["min_size"],
                                         max_net_size=self._train_config["max_size"],
                                         jitter=self._train_config["jitter"],
                                         shuffle=True)
        if len(valid_ann_fnames) > 0:
            valid_generator = BatchGenerator(valid_ann_fnames,
                                               self._train_config["valid_image_folder"],
                                               batch_size=self._train_config["batch_size"],
                                               labels=self._model_config["labels"],
                                               anchors=self._model_config["anchors"],
                                               min_net_size=self._model_config["net_size"],
                                               max_net_size=self._model_config["net_size"],
                                               jitter=False,
                                               shuffle=False)
        else:
            valid_generator = None
        print("Training samples : {}, Validation samples : {}".format(len(train_ann_fnames), len(valid_ann_fnames)))
        return train_generator, valid_generator

    def create_evaluator(self, model):

        detector = self.create_detector(model)
        train_ann_fnames = self._get_train_anns()
        valid_ann_fnames = self._get_valid_anns()

        train_evaluator = Evaluator(detector,
                                    self._model_config["labels"],
                                    train_ann_fnames,
                                    self._train_config["train_image_folder"])
        if len(valid_ann_fnames) > 0:
            valid_evaluator = Evaluator(detector,
                                        self._model_config["labels"],
                                        valid_ann_fnames,
                                        self._train_config["valid_image_folder"])
        else:
            valid_evaluator = None
        return train_evaluator, valid_evaluator

    def get_train_params(self):
        learning_rate=self._train_config["learning_rate"]
        save_dname=self._train_config["save_folder"]
        num_epoches=self._train_config["num_epoch"]
        return learning_rate, save_dname, num_epoches

    def get_labels(self):
        return self._model_config["labels"]
    
    def _get_train_anns(self):
        ann_fnames = glob.glob(os.path.join(self._train_config["train_annot_folder"], "*.xml"))
        return ann_fnames

    def _get_valid_anns(self):
        ann_fnames = glob.glob(os.path.join(self._train_config["valid_annot_folder"], "*.xml"))
        return ann_fnames
