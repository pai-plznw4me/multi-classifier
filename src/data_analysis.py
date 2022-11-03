"""
data 분석을 위한 모듈
"""
from tqdm import tqdm
from preprocessing import get_roi_coord, get_roi_image
from utils import get_all_paths, search_img_paths, get_names, paths2imgs, plot_images, read_image
import numpy as np
import matplotlib.pyplot as plt

# 총 이미지 개 수
paths = get_all_paths('./datasets/resize_images_512')
img_paths = search_img_paths(paths)[:16]
n_imgs = len(img_paths)
print('이미지 개 수 : {}'.format(n_imgs))

# right, left 분류
rights = []
lefts = []
names = get_names(img_paths)
_ = [rights.append(name) if 'right' in name else lefts.append(name) for name in names]
n_right = len(rights)
n_left = len(lefts)
print('right : {}, left: {}'.format(n_right, n_left))

# image size
imgs = paths2imgs(img_paths)
hs, ws = [], []
for img in tqdm(imgs):
    hs.append(np.array(img).shape[0])
    ws.append(np.array(img).shape[1])
h_w_ratio = np.array(hs) / np.array(ws)

fig, axes = plt.subplots(1, 3)
fig.set_size_inches(20, 5)
axes[0].hist(hs)
axes[1].hist(ws)
axes[2].hist(h_w_ratio)

# axis 1 축 중앙값 비교
center_values = imgs[0][:, 256]
center_values = np.sum(center_values, axis=1)

values, index = np.unique(center_values, return_counts=True)
ind = np.where(values == 0)
assert 0 == values[ind]
zero_length = index[ind]
zero_length = int(zero_length / 2)

imgs[0] = imgs[0][zero_length: -zero_length, :]
grey_img = read_image(img_paths[0], 'grey')
color_img = read_image(img_paths[0], 'rgb')
coord = get_roi_coord(grey_img)
cropped_img = get_roi_image(coord, color_img)
plot_images([cropped_img])

