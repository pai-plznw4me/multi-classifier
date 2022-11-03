import os
from tqdm import tqdm
from preprocessing import remove_noise
from utils import get_all_paths, search_img_paths, get_names, save_image

if __name__ == '__main__':
    # 지정된 폴더 내 이미지 파일만을 가져옵니다.
    folder_path = '/Users/seongjungkim/Desktop/marp/eyepacs'
    paths = get_all_paths(folder_path)
    img_paths = search_img_paths(paths)

    # 저장할 폴더 및 저장될 파일 경로를 생성합니다.
    names = get_names(img_paths)
    folder_name = 'remove_noise'
    save_dir = './datasets/{}'.format(folder_name)
    os.makedirs(save_dir, exist_ok=True)
    dst_paths = [os.path.join(save_dir, name) for name in names]

    # config
    image_size = 256

    # 사진 내 노이즈를 제거 후 저장합니다.
    error_record_path = './datasets/{}/errors.txt'.format(folder_name)
    f = open(error_record_path, 'w')
    for index, path in tqdm(enumerate(img_paths)):
        try:
            dst_path = dst_paths[index]

            # fundus 이미지에서 side noise 부분을 제거합니다.
            image = remove_noise(path, 5)
            image = image.astype('uint8')

            # 이미지를 저장합니다.
            save_image(image, dst_path)

        except Exception as e:
            print(e)
            f.write(str(dst_path + '\n'))
            f.flush()
    f.close()
