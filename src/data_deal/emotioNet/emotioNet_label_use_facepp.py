import sys
sys.path.append('..')
import os
import shutil

import glob

from utils.facepp_python.facepp import FacePp
from utils import common

# init face ++ sdk
face_attr_handle = FacePp()

IMG_ROOT_DIR = '/home/ruiming/workspace/pro/facialExpression/data/org/emotioNet/emotioNet_URLs_Download/downloader_beta1.0/IMAGES'
OUT_ROOT_DIR = '/train_tmp/fer/out/emotioNet/facepp'
if not os.path.exists(OUT_ROOT_DIR):
    os.makedirs(OUT_ROOT_DIR)

subdirs = os.listdir(IMG_ROOT_DIR)
subdirs.sort()

for subdir in subdirs:
    if 'dataFile' not in subdir:
        continue
    subdir_path = os.path.join(IMG_ROOT_DIR, subdir)
    out_subdir_path = os.path.join(OUT_ROOT_DIR, subdir)
    if not os.path.exists(out_subdir_path):
        os.makedirs(out_subdir_path)

    imgs = os.listdir(subdir_path)
    for idx,img in enumerate(imgs):
        src_img_path = os.path.join(subdir_path, img)

        # check if is image
        if not common.is_image(src_img_path):
            continue

        # copy image to dst
        dst_img_path = os.path.join(out_subdir_path, img)
        if os.path.exists(dst_img_path):
            continue
        shutil.copy(src_img_path, dst_img_path)

        # get and save face attribute from facepp
        face_attr_infos = face_attr_handle.get_face_attr(dst_img_path)
        if face_attr_infos is None:
            continue
        dst_json_file_path = os.path.join(out_subdir_path, img.rsplit('.', maxsplit=1)[0]+'.json')
        face_attr_handle.save_json(face_attr_infos, dst_json_file_path)

        if idx%100 == 0:
            print("{}:{}".format(subdir, idx))

