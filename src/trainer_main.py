from src.dataprovider import Dataprovider
from src.trainer import TrainClassifier


if __name__ == '__main__':
    # Dataprovider 셋팅
    input_dir = '../datasets/sanitizer_dough_256'
    label_path = '../datasets/sanitizer_dough.csv'  # '../datasets'
    batch_size = 64
    preprocess_input = None
    n_classes = 2
    tdp = Dataprovider(input_dir, label_path, batch_size, preprocess_input, n_classes, std=False)

    train_df, test_df = tdp.split_datasets(0.25)
    tdp.copy_datasets('train', train_df)
    tdp.copy_datasets('test', test_df)
    tdp.over_sampling()

    train_dp = Dataprovider('train', 'train', 60, None, 2)
    test_dp = Dataprovider('test', 'test', 60, None, 2)

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
    tc.set_model((299, 299, 3))
    tc.set_hparam()
    tc.training(train_dp, test_dp)
