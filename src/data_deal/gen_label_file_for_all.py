import os
from utils import common

DATASET_ROOT_DIR = '/train_tmp/fer/select/finish2'
OUT_FILE_PATH = '/train_tmp/fer/select/finish2/train.txt'

phase = 'train'
datasets = os.listdir(DATASET_ROOT_DIR)

with open(OUT_FILE_PATH, 'w') as fp:
    for dataset in datasets:
        dataset_dir = os.path.join(DATASET_ROOT_DIR, dataset)
        if not os.path.isdir(dataset_dir):
            continue

        phase_dir = os.path.join(dataset_dir, phase)
        if not os.path.exists(phase_dir) or not os.path.isdir(phase_dir):
            continue

        #cats_map = {'Angry':0, 'Happy':1, 'Neutral':2, 'Sad':3}
        cats_map = {'angry_heavy':0, 'angry_light':0, 'happy':1, 'neutral':2, 'sad_heavy':3, 'sad_light':3}

        for cat in cats_map:
            cat_dir = os.path.join(phase_dir, cat)
            if not os.path.exists(cat_dir) or not os.path.isdir(cat_dir):
                print("Warning! {} have no {}".format(dataset, cat))
                continue

            imgs = os.listdir(cat_dir)
            for img in imgs:
                if not common.is_image(img):
                    continue

                # get face rectangle
                face_rect_file_path = os.path.join(cat_dir, img.rsplit('.', maxsplit=1)[0]+'.facerect')
                if not os.path.exists(face_rect_file_path):
                    continue
                rects = common.read_lists_from_txtfile(face_rect_file_path)
                if len(rects)<1:
                    continue
                rect = rects[0]
                rect = [int(x) for x in rect[:4]]

                fp.write('{};{};{};{};{};{}\n'.format(os.path.join(dataset, phase, cat, img), cats_map[cat], rect[0], rect[1], rect[2], rect[3]))
        print("{} dealed!".format(dataset))