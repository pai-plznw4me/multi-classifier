import os
from tqdm import tqdm
from preprocessing import isometrically_resizes, resize_by_large, pads
from utils import get_all_paths, search_img_paths, get_names, read_image, save_image


if __name__ == '__main__':
    # 지정된 폴더 내 이미지 파일만을 가져옵니다.
    foldername = 'donut'
    folder_path = './datasets/{}'.format(foldername)
    paths = get_all_paths(folder_path)
    img_paths = search_img_paths(paths)

    # 저장할 폴더 및 저장될 파일 경로를 생성합니다.
    names = get_names(img_paths)
    dirname = 'sanitizer_dough_256'
    save_dir = './datasets/{}'.format(dirname)
    os.makedirs(save_dir, exist_ok=True)
    dst_paths = [os.path.join(save_dir, name) for name in names]

    # 이미지를 지정된 크기로 resize 한 후 저장합니다.
    error_record_path = './datasets/{}/errors.txt'.format(dirname)
    image_size = 256
    f = open(error_record_path, 'w')
    for index, path in tqdm(enumerate(img_paths)):
        try:
            img = read_image(path, 'rgb')
            dst_path = dst_paths[index]
            resized_imgs = isometrically_resizes([img], image_size, resize_by_large)
            padded_imgs = pads(resized_imgs, image_size, image_size, 'center')
            save_image(padded_imgs[0], dst_path)

        except Exception as e:
            f.write(str(e))
            f.flush()
    f.close()
