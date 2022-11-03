import shutil
import numpy as np
import os
import pandas as pd
from utils import get_all_paths, filter_img_paths, get_names, paths2np, plot_images, paths2imgs
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


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
        print('Epoch per steps : {}'.format(n_steps))
        return n_steps

    def __getitem__(self, idx):

        # 배치 파일 가져올 인덱스 슬라이스를 생성합니다.
        slice_ = slice(self.batch_size * idx, self.batch_size * (idx + 1))

        # 배치 단위로 입력 파일 경로를 가져옵니다. example) [a/a.jpeg, a/b.jpeg]
        batch_df = self.merged_df[slice_]
        batch_paths = batch_df.path.to_list()
        batch_xs = np.stack(paths2imgs(batch_paths), axis=0)

        # 배치 단위로 출력 라벨 정보를 가져옵니다. example) [0, 1]
        batch_ys = np.array(batch_df.label)
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

    def split_datasets(self, test_size):
        X_train, X_test, y_train, y_test = train_test_split(self.merged_df,
                                                            self.merged_df.label,
                                                            test_size=test_size, stratify=self.merged_df.label)
        return X_train, X_test

    def copy_datasets(self, save_dir, save_df):
        os.makedirs(save_dir, exist_ok=True)

        src_paths = save_df.path
        names = get_names(src_paths)
        dst_paths = list(map(lambda x: os.path.join(save_dir, x), names))

        for src_path, dst_path in zip(src_paths, dst_paths):
            shutil.copy(src_path, dst_path)

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

    def over_sampling(self):
        """
        Description:
            가장 많은 class code 를 가지고 있는 데이터 셋에 맞게 모든 항목들을 그 숫자에 맞게 복사합니다.
            inplace 형태로 변환합니다.
        Returns:
        """
        counts = self.merged_df.label.value_counts().values
        class_codes = self.merged_df.label.value_counts().index
        max_count = max(counts)
        print('가장 개 수:{} '.format(max_count))

        add_df_bucket = []
        add_df_bucket.append(self.merged_df)
        # 가장 개 수가 많은 class code 의 샘플 만큼 모든 class code 의 데이터를 램덤하게 추출해 추가합니다.
        for class_code, count in zip(class_codes, counts):
            # 추가할 데이터 개 수
            n_oversample = max_count - count

            # target class code 의 DataFrame 추출
            mask = self.merged_df.label.map(lambda x: int(x) == int(class_code))
            trgt_df = self.merged_df.loc[mask]

            # oversample 개 수 만큼 random 으로 추출
            rand_indices = np.random.choice(trgt_df.index.values, n_oversample)
            add_df = trgt_df.loc[rand_indices]
            add_df_bucket.append(add_df)

        # oversample 된 dataframe 을 기존 anno df 로 치환
        self.merged_df = pd.concat(add_df_bucket)


if __name__ == '__main__':
    input_dir = '../datasets/sanitizer_dough_256'
    label_path = '../datasets/sanitizer_dough.csv'  # '../datasets'
    batch_size = 60
    preprocess_input = None
    n_classes = 2
    td = Dataprovider(input_dir, label_path, batch_size, preprocess_input, n_classes)
    paths = td.merged_df.path
    xs, ys = td[1]
    plot_images(xs)
