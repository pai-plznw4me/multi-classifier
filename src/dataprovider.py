import numpy as np
import os
import pandas as pd
from utils import get_all_paths, filter_img_paths, get_names, fill_df, save_np, paths2np, fill_np, plot_images
from preprocessing import remove_paddings_and_resizes
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical


class Dataprovider(Sequence):
    """
    Description:
        지정된 이미지 폴더내 이미지 파일들을 읽어와 필요한 정보를  추출해 통합된 하나의 DataFrame을 제공합니다.
        제공되는 DataFrame 은 아래와 같은 구조를 가지고 있습니다.
        DataFrame 구조
            'filename' : 이미지 파일 이름
            'label' : 라벨
            'path' : 이미지 저장 경로
            'preproc' : preprocessing 된 파일의 저장 경로

        데이터를 읽어와 batch size 만큼 잘라 제공합니다. keras fit 에 적용할수 있습니다.

        처음 generator 가 작동시에는 필수 전처리 후 지정된 폴더에 저장됩니다.
        'input_dir'/tmp 폴더에 저장됩니다.
        하지만 다음 epoch 에는 저장된 기 전처리 된 데이터를 읽어와 사용해 속도가 빠릅니다.

        이미지를 읽다가 에러가 나는 경우 해당 데이터를 사용하지 않습니다.
        에러난 데이터는 error.txt 에서 확인 가능합니다.

    Args:
        :param str input_dir:
        :param str label_path:
        :param str batch_size:
        :param str preprocess_input:
        :param str n_classes:
        :param str shuffle=True:
        :param str std=True:

    Usage:
        input_dir = '../datasets/resize_and_pad_256x256'
        label_path = '../datasets/trainLabels.csv'  # '../datasets'
        save_dir = '../datasets/Preproc'
        batch_size = 3
        preprocess_input = None
        n_classes = 2
        td = Dataprovider(input_dir, output_dir, batch_size, preprocess_input, n_classes)
        paths = td.merged_df.path
        xs, ys = td[1]

    """

    def __init__(self, input_dir, label_path, batch_size, preprocess_input, n_classes, shuffle=True, std=True):
        # 인스턴스화
        self.batch_size = batch_size
        self.preprocess_input = preprocess_input
        self.label_path = label_path
        self.input_dir = input_dir
        self.shuffle = shuffle
        self.std = std
        self.n_classes = n_classes
        self.save_dir = os.path.join(input_dir, 'tmp')

        # 전처리 된 파일을 저장할 저장 폴더를 지정합니다.
        os.makedirs(self.save_dir, exist_ok=True)
        self.error_data_path = os.path.join(self.input_dir, 'error.txt')

        # 출력 데이터
        self.label_df = pd.read_csv(self.label_path)

        # 입력 데이터
        self.input_paths = get_all_paths(self.input_dir)
        self.input_paths = filter_img_paths(self.input_paths)
        print('총 데이터 개 수 : {}'.format(len(self.input_paths)))
        self.input_names = get_names(self.input_paths, ext=True)

        # 입력 데이터 Dataframe과 출력 데이터 Dataframe 을 하나로 합침니다.
        self.merged_df = self.label_df.copy()

        # 전처리가 된 column 을 생성합니다.
        self.merged_df['preproc'] = None

        # 입력 ,출력 데이터를 DataFrame 으로 통합
        self.merged_df['path'] = self.merged_df['filename'].map(lambda x: os.path.join(self.input_dir, x))

        # 존재 하지 않는 이미지 파일 목록을 merged_df 에서 제거 합니다.
        if len(self.label_df) != len(self.input_paths):
            masked_df, unmasked_df, mask = self.data_sync()
            self.merged_df = masked_df

        if shuffle:
            self.merged_df = self.merged_df.sample(frac=1)

    def __len__(self):
        """
        Description:
            step 개 수를 생성합니다.
        :return:
        """
        n_steps = np.floor(len(self.merged_df) / self.batch_size).astype(np.int)
        print('1 Epoch : {}'.format(n_steps))
        return n_steps

    def __getitem__(self, idx):

        # 배치 파일 가져올 인덱스 슬라이스를 생성합니다.
        slice_ = slice(self.batch_size * idx, self.batch_size * (idx + 1))

        # 배치 단위로 입력 파일 경로를 가져옵니다. example) [a/a.jpeg, a/b.jpeg]
        batch_df = self.merged_df[slice_]

        # 배치 단위로 출력 라벨 정보를 가져옵니다. example) [0, 1]
        batch_ys = np.array(batch_df.label)

        # preproc 에서 None 대신 preprocessing 된 경로가 모두 들어있으면 아래 코드가 수행 됩니다.
        if not batch_df.preproc.isna().all():
            batch_xs = paths2np(batch_df.preproc)

            # batch 사이즈 보다 개 수가 작으면 batch_xs[0] 데이터를 복사해 batch_xs 에 추가합니다.
            batch_xs = fill_np(self.batch_size - len(batch_ys), batch_xs)
            batch_ys = fill_np(self.batch_size - len(batch_ys), batch_ys)

        # preproc column 에서 None 이 하나라도 들어 있으면 아래 코드가 수행 됩니다.
        else:
            batch_paths = batch_df.path

            # 이미지를 299x299x3 형태로 출력합니다. (※ side padding  을 제거하고 isometrically resize 합니다.)
            # 이미지 전처리시 문제가 생기면 에러가 발생된 인덱스를 error_ind 에 담아 반환합니다.
            batch_xs, error_ind = remove_paddings_and_resizes(batch_paths, 299)

            if error_ind:
                # error_ind 에 해당하는 부분을 batch_ys 에서 제거 합니다.
                batch_ys = pd.Series(batch_ys).drop(error_ind)

                # batch df 에서 에러가 난 row 을 제거 합니다.
                batch_df.drop(batch_df.iloc[error_ind].index, inplace=True)

                # error_ind 개 수 만큼 batch_xs[0] 데이터를 복사해 batch_xs 에 추가합니다.
                batch_xs = fill_np(len(error_ind), batch_xs)
                batch_ys = fill_df(len(error_ind), batch_ys)
                batch_df = fill_df(len(error_ind), batch_df)

                # 전체 데이터에서 에러가 난 데이터를 삭제합니다. 그리고 삭제한 데이터 로그를 저장합니다.
                batch_error_paths = batch_paths.iloc[error_ind]
                batch_error_index = batch_error_paths.index
                self.merged_df.drop(batch_error_index, inplace=True)

                # 에러가 난 파일 경로를 저장합니다.
                f = open(self.error_data_path, 'a')
                for batch_error_path in batch_error_paths:
                    f.write(batch_error_path + '\n')
                f.close()

            batch_xs = batch_xs.astype(float)
            # backbone에 따른 input data preprocess 진행
            if self.preprocess_input:
                batch_xs = self.preprocess_input(batch_xs)

            # input data std 진행
            if self.std:
                batch_xs = self.standardize(batch_xs)

            # preprocessing 이 된 numpy 파일을 저장합니다.
            names = get_names(batch_df.path, ext=False)
            dst_paths = [os.path.join(self.save_dir, name) + '.npy' for name in names]
            save_np(dst_paths, batch_xs)

            # preprocessing 된 이미지 경롤를 저장합니다.
            self.merged_df.loc[batch_df.index, 'preproc'] = dst_paths
            batch_ys = to_categorical(batch_ys, self.n_classes)

        return batch_xs, batch_ys

    @staticmethod
    def standardize(xs):
        for ind, x in enumerate(xs):
            x = x - np.mean(x, axis=(0, 1))
            x = x / np.std(x)
            xs[ind] = x
        return xs

    def on_epoch_end(self):
        if self.shuffle:
            self.merged_df = self.merged_df.sample(frac=1).reset_index(drop=True)

    def data_sync(self):
        """
        Description:
            입력 이미지와 라벨 데이터 간의 sync 을 맞춥니다.
            label df 에는 존재하지만 실제 이미지 파일이 존재하지 않다면 해당 경로를 삭제 합니다.

            실제 존재한 파일: a.jpg, c.jpg
            label_df
            a.jpg   a.jpg
            b.jpg → c.jpg
            c.jpg

        :return:
        """
        mask = self.merged_df.filename.isin(self.input_names)
        masked_df = self.merged_df.loc[mask]
        unmasked_df = self.merged_df.loc[~mask]
        return masked_df, unmasked_df, mask


if __name__ == '__main__':
    input_dir = '../datasets/resize_and_pad_256x256'
    label_path = '../datasets/trainLabels.csv'  # '../datasets'
    batch_size = 3
    preprocess_input = None
    n_classes = 4
    td = Dataprovider(input_dir, label_path, batch_size, preprocess_input, n_classes)
    paths = td.merged_df.path
    xs, ys = td[1]
    plot_images(xs)
