import math
import os
import cv2
import numpy as np
from tqdm import tqdm

from utils import read_image
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import random


def paths2imgs(paths, resize=None, error=None):
    """
    Description:
    :param paths: list, [str, str, ... str], 이미지 저장 경로
    :param resize: tuple, (h, w)
    :param error: 잘못된 경로를 반환합니다.
    :return:
    """
    imgs = []
    error_paths = []
    for path in paths:
        try:
            img = path2img(path, resize)
            imgs.append(img)
        except:
            print(os.path.exists(path))
            print("{} 해당 경로에 파일이 존재하지 않습니다.".format(path))
            error_paths.append(path)

    if error == 'error_return':
        return imgs, error_paths

    return imgs


def path2img(path, resize=None):
    """
    Description:
    경로에 있는 이미지를 RGB 컬러 형식으로 불러옵니다
    resize 에 값을 주면 해당 크기로 이미지를 불러옵니다.
    :param path: str
    :param resize: tuple or list , (W, H)
    return img, ndarray, shape=(h, w, 3)
    """

    # 경로 중 한글명이 있어 cv2 만으로는 읽을 수 없었기 때문에, numpy 로 파일을 읽은 후 이를 cv2.imdecode 를 통해 이미지로 변환합니다.
    img = cv2.imread(path)

    # BGR 을 RGB 로 변환합니다.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if resize:
        h, w = resize
        img = cv2.resize(img, (w, h))

    return


def plot_images(imgs, names=None, random_order=False, savepath=None):
    h = math.ceil(math.sqrt(len(imgs)))
    fig = plt.figure()
    plt.gcf().set_size_inches((20, 20))
    for i in range(len(imgs)):
        ax = fig.add_subplot(h, h, i + 1)
        if random_order:
            ind = random.randint(0, len(imgs) - 1)
        else:
            ind = i
        img = imgs[ind]
        plt.axis('off')
        plt.imshow(img)
        #
        if not names is None:
            ax.set_title(str(names[ind]))
    if not savepath is None:
        plt.savefig(savepath)
    plt.tight_layout()
    plt.show()


def pads(images, height, width, location='random'):
    """
    Description:
    Args:
        images: ndarray, shape = (N, H, W, C) or (N, H, W)
        height: int, output array 의 height size
        width: int, output array 의 width size
        location:
            random: image 을 최종 출력 array random 한 위치에 둠
            center: image 을 최종 출력 array 중심에 둠
    Returns: imgs, ndarray, shape=(N, height, width, C) or (N, height, width)
    """
    if location == 'random':
        seq = iaa.Sequential([
            iaa.PadToFixedSize(width=width, height=height)])
        imgs = seq(images=images)
    elif location == 'center':
        # 이미지를 중앙에 위치하게 합니다.
        imgs = []
        for image in images:
            assert len(image.shape) == 3, '이미지는 color 이미지여야 합니다.'
            img_h, img_w, img_ch = image.shape
            background = np.zeros((height, width, img_ch)).astype(int)

            # 이미지가 놓일 시작 위치를 찾습니다.
            h_gap = int((height - img_h) / 2)
            w_gap = int((width - img_w) / 2)

            background[h_gap:h_gap + img_h, w_gap:w_gap + img_w] = image
            # 이미지가 놓일 시작 위치를 찾습니다.
            imgs.append(background)
    else:
        raise NotImplementedError
    imgs = np.array(imgs)
    return imgs


def isometrically_resizes(images, target_side_size, resize_fn):
    """
    Description:
        이미 불러온 이미지들을 아래와 같은 방법으로 resize 합니다.
        이미지의 비율을 유지한 체 resize 합니다.
        이미지 내 가장 크기가 작은 쪽은 s_size에 맞춘 후 반홥합니다.

    Usage:
            # 지정된 폴더 내 이미지 파일만을 가져옵니다.
        folder_path = './images'
        paths = get_all_paths(folder_path)
        img_paths = search_img_paths(paths)

        # 저장할 폴더 및 저장될 파일 경로를 생성합니다.
        names = get_names(img_paths)
        save_dir = './tmp'
        os.makedirs(save_dir, exist_ok=True)
        dst_paths = [os.path.join(save_dir, name) for name in names]

        for index, path in tqdm(enumerate(img_paths)):
            img = read_image(path, 'rgb')
            dst_path = dst_paths[index]
            resized_imgs = isometrically_resizes([img], 256, resize_by_large)
            Image.fromarray(resized_imgs[0]).save(dst_paths[index])

    :param images: 들어오는 도면 이미지들
    :param resize_fn: int, 이미지 resize 을 수행하는 function
    :param target_side_size: int, 반환 할 이미지 내 가장 크기가 작은 면의 크기 또는 가장 큰 크기
    """
    resized_images = []
    for ind, image in enumerate(images):
        resized_images.append(resize_fn(image, target_side_size))
    return resized_images


def resize_by_large(image, l_size):
    """
    Description:
    비율이 보전 된 상태로 주어진 resize 합니다.
        :param ndarray image: shape (h, w)
        :param l_size: 이미지내 비율이 가작 큰 면(side)

    :return ndarray image:
    """
    h, w = image.shape[:2]
    max_side = np.maximum(h, w)
    l_ratio = l_size / max_side
    resized_h = int(h * l_ratio)
    resized_w = int(w * l_ratio)
    image = cv2.resize(image, dsize=(resized_w, resized_h))
    return image


def get_roi_coord(image):
    """
    Description:
        fundus image 내 noise 영역 위치를 찾아 제공합니다.
        noise 에 대한 정의 내용은 marp fundus dataset 내 wiki 을 참조하세요.
    :param ndarray image: shape (h, w)
    :return list regions: [[axis 0 min index, axis 0 max index ], [axis 1 min index axis 1 max index ]]
    """
    h, w = image.shape[:2]

    center_h_index = int(h / 2)
    center_ws = image[center_h_index, :]

    width_index = []
    height_index = [0, h]

    for index, value in enumerate(center_ws):
        if value != 0:
            width_index.append(index)
            break

    for index, value in enumerate(center_ws[::-1]):
        if value != 0:
            width_index.append(w - index)
            break

    return [height_index, width_index]


def get_roi_image(coord, image):
    roi = image[coord[0][0]:coord[0][1], coord[1][0]:coord[1][1]]
    return roi


def remove_noise(path, margin=2):
    """
    Description:
        fundus image 내 noise 영역을 제거 합니다.

    Usage:
        # 지정된 폴더 내 이미지 파일만을 가져옵니다.
        folder_path = './images'
        paths = get_all_paths(folder_path)
        img_paths = search_img_paths(paths)

        # 저장할 폴더 및 저장될 파일 경로를 생성합니다.
        names = get_names(img_paths)
        save_dir = './tmp'
        os.makedirs(save_dir, exist_ok=True)
        dst_paths = [os.path.join(save_dir, name) for name in names]

        # 사진 내 노이즈를 제거 후 저장합니다.
        for index, path in tqdm(enumerate(img_paths)):
            dst_path = dst_paths[index]
            remove_noise(path, dst_path)

    :param str path:
    :param str save_path:
    :return ndarray roi_image:
    """
    grey_img = read_image(path, 'grey')
    grey_img = (grey_img.astype('int') - margin)
    grey_img = np.clip(grey_img, 0, 255).astype('uint')
    color_img = read_image(path, 'rgb')
    coord = get_roi_coord(grey_img)
    roi_image = get_roi_image(coord, color_img)

    return roi_image


def remove_paddings_and_resizes(paths, image_size, margin=5):
    """
    Description:
        복수개의 fundus 이미지 저장 경로를 입력받아 fundus 주변의
        left right side padding 정보를 제거하고 지정된 정방형 크기(ex 299x299)로 변환합니다.
    Usage:
        dirname = '/Users/seongjungkim/Desktop/marp/eyepacs/images'
        paths = get_all_paths(dirname)[:100]
        imgs = remove_paddings_and_resizes(paths, image_size=299)
        imgs = plot_images(imgs)

    :param list paths: 이미지 저장 경로
    :param int image_size: 최종 이미지 크기
    :param int margin: 모든 pixel 값에서 margin 값을 뺍니다.
    :return ndarray padded_imgs:
    """
    images = []
    error_indices = []
    for ind, path in tqdm(enumerate(paths)):
        try:
            image = remove_noise(path, margin)
            image = image.astype('uint8')
            images.append(image)
        except OSError as ose:
            print('Error 발생 : {}'.format(str(ose)))
            error_indices.append(ind)

    resized_imgs = isometrically_resizes(images, image_size, resize_by_large)
    padded_imgs = pads(resized_imgs, image_size, image_size, 'center')

    return padded_imgs, error_indices

