import os
import glob
import shutil
import random

IMG_ROOT_DIR = r'G:\pro\facialexpression\data\youtube\expression\Neutral'
OUT_IMG_ROOT_DIR = r'G:\pro\facialexpression\data\youtube\expression\Neutral_toselect'
if not os.path.exists(OUT_IMG_ROOT_DIR):
    os.makedirs(OUT_IMG_ROOT_DIR)

imgs = os.listdir(IMG_ROOT_DIR)

infos = {}
for img in imgs:
    person = img.rsplit('_', maxsplit=1)[0]
    if person not in infos:
        infos[person] = []
    infos[person].append(img)

for p in infos:
    imgs = infos[p]
    # remove first 2 images
    if len(imgs) > 10:
        imgs = random.sample(imgs, 10)

    for img in imgs:
        src_img_path = os.path.join(IMG_ROOT_DIR, img)
        dst_img_path = os.path.join(OUT_IMG_ROOT_DIR, img)
        shutil.copy(src_img_path, dst_img_path)
        os.remove(src_img_path)

#print(infos)