import math
import os
import re
from random import random
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image


def show_fundus(df, dr_level, n_samples, **kwargs):
    trgt_df = df.loc[df.level == dr_level]
    paths = trgt_df.iloc[:n_samples].path.values.tolist()
    imgs = paths2imgs(paths)
    plot_images(imgs, **kwargs)


def plot_images(imgs, names=None, random_order=False, savepath=None, fig_size=(20, 20)):
    h = math.ceil(math.sqrt(len(imgs)))
    fig = plt.figure()
    plt.gcf().set_size_inches(fig_size)
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


def path2np(path):
    """
    Description:
        경로에 있는 numpy 을 부러옵니다.
        :param path: str
        return np, ndarray, shape=(h, w, 3)
    """
    return np.load(path)


def paths2nps(paths):
    nps = []
    for path in paths:
        nps.append(path2np(path))
    return nps


def paths2np(paths):
    nps = []
    for path in paths:
        nps.append(path2np(path))
    return np.array(nps)


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

    return img


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


def get_names(paths, ext=True):
    if ext:
        return [os.path.split(path)[-1] for path in paths]
    else:
        return [os.path.splitext(os.path.split(path)[-1])[0] for path in paths]


def save_imgs(dst_paths, src_imgs):
    """
    :param dst_paths: list = [str, str, ... str]
    :param src_imgs: ndarray
    :return:
    """
    for path, img in tqdm(zip(dst_paths, src_imgs)):
        Image.fromarray(img).save(path)


def get_all_paths(folder_path):
    """
    Description:
        folder 내 모든 파일 정보를 가져옵니다.
    Args:
        :param str folder:
    :return:
    """
    all_file_paths = []
    for folder, sub_folders, files in os.walk(folder_path):
        file_paths = [os.path.join(folder, file) for file in files]
        all_file_paths.extend(file_paths)
    return all_file_paths


def search_img_paths(paths):
    """
    Description:
        입력 경로중 '.gif|.jpg|.jpeg|.tiff|.png' 해당 확장자를 가진 파일만 반환합니다.

    Args:
        :param: paths, [str, str, str ... str]

    :return: list, [str, str, str ... str]
    """
    regex = re.compile("(.*)(\w+)(.gif|.jpg|.jpeg|.tiff|.png|.bmp|.JPG|.JPEG)")
    img_paths = []
    for path in paths:
        if regex.search(path):
            img_paths.append(path)
    return img_paths


def filter_img_paths(paths):
    """
    Description:
        입력 경로중 '.gif|.jpg|.jpeg|.tiff|.png' 해당 확장자를 가진 파일만 반환합니다.

    Args:
        :param: paths, [str, str, str ... str]

    :return: list, [str, str, str ... str]
    """
    regex = re.compile("(.*)(\w+)(.gif|.jpg|.jpeg|.tiff|.png|.bmp|.JPG|.JPEG)")
    img_paths = []
    for path in paths:
        if regex.search(path):
            img_paths.append(path)
    return img_paths


def read_image(path, color):
    """
    Description:
    이미지를 경로에서 불러옵니다.

    :param str path:
    :param str color:
    :return ndarray img: shape (h, w, ch ) or (h, w)
    """
    img = Image.open(path)
    if color == 'grey':
        img = img.convert('L')
    elif color == 'rgb':
        img = img.convert('RGB')
    else:
        raise NotImplementedError
    img = np.array(img)
    return img


def save_image(image, save_path):
    """
    이미지를 저장합니다.

    :param ndarray image:
    :param str save_path:
    :return:
    """
    image = image.astype('uint8')
    Image.fromarray(image).save(save_path)


def fill_np(n_fill, xs):
    """
    Description:
        batch size 보다 xs 의 개수가 부족하면 xs[0] 데이터를 복사해
        xs 에 추가해 batch_size 만큼 xs 개 수를 채워 반환합니다.

        example)
        xs = [[1,1]  batch_size =5
             [1,2]
             [1,3]]

        return
        xs = [[1,1]  batch_size =5
             [1,2]
             [1,3]
             [1,1]
             [1,1]]

    Usage:
        xs = np.arange(0, 100).reshape(50, 2)
        ys = np.arange(53)
        error_indices = [0, 1, 2]
        xs, ys = fill(error_indices, xs, ys)

    Args:
        :param int n_fill: 채워야 할 개 수
        :param ndarray xs: shape=(N, h, w, ch)
        :param DataFrame ys: shape=(N, h, w, ch)
        :param DataFrame df: shape=(N, h, w, ch)
    """
    # 채워야 할 개 수가 0이 아니면 아래 코드를 수행 합니다.
    assert type(n_fill) is int and (n_fill > 0 or n_fill == 0)
    if n_fill:
        xs = np.concatenate([xs, np.array([xs[0]] * n_fill)], axis=0)
    return xs


def fill_df(n_fill, df):
    assert type(n_fill) is int and (n_fill > 0 or n_fill == 0)
    # 채워야 할 개 수가 0이 아니면
    if n_fill:
        copied_df = [df.iloc[0:1]] * n_fill
        df = pd.concat([df, *copied_df], axis=0)
    return df


def save_np(dst_paths, src_nps):
    """
    :param dst_paths: list = [str, str, ... str]
    :param src_imgs: ndarray
    :return:
    """
    for path, src_np in zip(dst_paths, src_nps):
        np.save(path, src_np)


def generate_label(type, save_path, **kwargs):
    """
    Description:
    label 데이터을 생성합니다.
    만약 폴더가 2가지로 나뉘어져 있고 A, B 폴더에 있는 데이터를 각각 라벨 0,1 로 나누고 싶다면
    아래와 같이 사용 할 수 있습니다.

    A
    |-a.jpg
    |-b.jpg

    B
    |-c.jpg
    |-d.jpg

    result:
        result.txt
        a.jpg 0
        b.jpg 0
        c.jpg 1
        d.jpg 1

    Usage:
        generate_label(type='name', save_path= save/path/result.txt, path/to/A=0, path/to/B=1)

    :return: none
    """

    total_paths, total_names, total_labels = [], [], []
    for folder_path, label in kwargs.items():

        # 이미지 경로를 가져옵니다.
        paths = get_all_paths(folder_path)
        assert len(paths) > 0, '지정된 폴더에 파일이 존재하지 않습니다.'
        paths = filter_img_paths(paths)
        assert len(paths) > 0, '지정된 폴더에 이미지 파일이 존재하지 않습니다.'

        total_names.extend(get_names(paths))
        total_labels.extend([label] * len(paths))
        total_paths.extend(paths)

    # DataFrame 화 합니다.
    if type == 'filename':
        df = pd.DataFrame({type: total_names, 'labels': total_labels})
    elif type == 'path':
        df = pd.DataFrame({type: total_paths, 'labels': total_labels})
    else:
        print('name or path 을 입력해야 합니다.')
        raise NotImplementedError

    if save_path:
        df.to_csv(save_path, index=None)

    return df


if __name__ == '__main__':
    a = {'/Users/kimseongjung/PycharmProjects/MARP-dataset-eyepacs/eyepacs/datasets/1234567': 0,
         '/Users/kimseongjung/PycharmProjects/MARP-dataset-eyepacs/eyepacs/datasets/9876543': 1}
    generate_label('filename', '../tmp.csv', **a)
