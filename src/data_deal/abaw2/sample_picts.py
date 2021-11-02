import sys
import os
import shutil
import glob

IMG_ROOT_DIR = r'G:\pro\facialexpression\data\abaw2\expression\Sad'
imgs = glob.glob(IMG_ROOT_DIR+r'\*.jpg')

OUT_IMG_DIR = r'G:\pro\facialexpression\data\abaw2\expression\Sad_select'
if not os.path.exists(OUT_IMG_DIR):
    os.makedirs(OUT_IMG_DIR)

for idx,img in enumerate(imgs):
    if idx % 20 == 0:
        dst_img_path = os.path.join(OUT_IMG_DIR, os.path.basename(img))
        shutil.copy(img, dst_img_path)

    if idx%1000 == 0:
        print("{}/{}".format(idx, len(imgs)))