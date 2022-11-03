import os
from src.dataprovider import Dataprovider
from src.trainer import TrainClassifier
from src.utils import generate_label


if __name__ == '__main__':
    # Dataprovider 셋팅
    input_dir = '../datasets/sanitizer_dough_256'
    label_path = '../datasets/sanitizer_dough.csv'  # '../datasets'
    batch_size = 64
    preprocess_input = None
    n_classes = 2
    tdp = Dataprovider(input_dir, label_path, batch_size, preprocess_input, n_classes, std=False)

    # Trainer 셋팅
    hparam = {"lr": 0.001,
              "log_dir": "../logs/0",
              "model_dir": "../models/0",
              "opt": "momentum",
              "preproc": True,
              "backbone_name": 'resnet50',
              "pretrain": True,
              "freezing": True,
              "n_epochs": 2,
              "step_per_epoch": 1,
              "n_classes": 2}

    tc = TrainClassifier(**hparam)
    tc.set_model((256, 256, 3))
    tc.set_hparam()
    tc.training(tdp, tdp)
