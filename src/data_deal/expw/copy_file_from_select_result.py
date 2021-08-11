import sys
sys.path.append('..')
import os
import glob
import shutil

from utils import common

ORG_IMG_ROOT_DIR = '/train_tmp/fer/out/FacialExpressionOrignialData/ExpW/train'
SELECTED_IMG_ROOT_DIR = '/home/ruiming/workspace/pro/train_tmp/fer/select/finish2/ExpW/train'
OUT_IMG_ROOT_DIR =      '/home/ruiming/workspace/pro/train_tmp/fer/select/finish2/ExpW/train_s'

# get all image path and name of
org_images_path = glob.glob(os.path.join(ORG_IMG_ROOT_DIR, '*/*.jpg'))
org_images_names = [os.path.basename(x) for x in org_images_path]

# what we have?
cats_names = os.listdir(SELECTED_IMG_ROOT_DIR)

for cat_name in cats_names:
    cat_dir = os.path.join(SELECTED_IMG_ROOT_DIR, cat_name)
    if not os.path.isdir(cat_dir):
        continue

    out_cat_dir = os.path.join(OUT_IMG_ROOT_DIR, cat_name)
    if not os.path.exists(out_cat_dir):
        os.makedirs(out_cat_dir)

    images = os.listdir(cat_dir)
    for img in images:
        if not common.is_image(img):
            continue
        # copy image and rectfile
        if img not in org_images_names:
            print("Warning ! {} not in ORG_IMG_ROOT_DIR")
            continue
        src_img_path = org_images_path[org_images_names.index(img)]
        dst_img_path = os.path.join(out_cat_dir, img)

        src_facerect_path = src_img_path.rsplit('.', maxsplit=1)[0]+'.facerect'
        if not os.path.exists(src_facerect_path):
            print("Have image but no rectangle!")
            continue
        dst_facerect_path = os.path.join(out_cat_dir, os.path.basename(src_facerect_path))

        shutil.copy(src_facerect_path, dst_facerect_path)
        shutil.copy(src_img_path, dst_img_path)