import numpy as np

from dataprovider import Dataprovider
from src.preprocessing import plot_images

input_dir = '../datasets/sanitizer_dough_256'
label_path = '../datasets/sanitizer_dough.csv'  # '../datasets'
batch_size = 64
preprocess_input = None
n_classes = 2
td = Dataprovider(input_dir, label_path, batch_size, preprocess_input, n_classes, std=False)
paths = td.merged_df.path
xs, ys = td[1]

plot_images(xs.astype('uint'), ys)

