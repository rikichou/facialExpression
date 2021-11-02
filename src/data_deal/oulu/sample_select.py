import os
import glob
import shutil
import random

IMG_ROOT_DIR = r'E:\workspace\pro\facialExpression\data\select_finish2_rename\Oulu_CASIA_NIR_VIS\train\sad_light'
OUT_IMG_ROOT_DIR = r'E:\workspace\pro\facialExpression\data\select_finish2_rename\Oulu_CASIA_NIR_VIS\train\sad_light_sampled'
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
    if len(imgs) > 2:
        imgs = imgs[2:]
    if len(imgs) > 5:
        imgs = random.sample(imgs, 5)

    for img in imgs:
        src_img_path = os.path.join(IMG_ROOT_DIR, img)
        dst_img_path = os.path.join(OUT_IMG_ROOT_DIR, img)
        shutil.copy(src_img_path, dst_img_path)

#print(infos)