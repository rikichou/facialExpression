import os
import random
import shutil
import glob

SAMPLES_PER_VIDEO = 5
IMG_ROOT_DIR = r'/home/ruiming/workspace/pro/facialExpression/data/select/finish2/streamax_0825/train/happy'
OUT_IMG_ROOT_DIR = r'/home/ruiming/workspace/pro/facialExpression/data/select/streamax/train/happy'
if not os.path.exists(OUT_IMG_ROOT_DIR):
    os.makedirs(OUT_IMG_ROOT_DIR)

imgs = glob.glob(IMG_ROOT_DIR+'/*.jpg')

statistics = {}
for img in imgs:
    name = os.path.basename(img)
    v = name[:5]

    if v not in statistics:
        statistics[v] = []

    statistics[v].append(img)

items = statistics.items()
soted = sorted(items, key=lambda x:len(x[1]), reverse=True)

print("Total {} images, {} videos, max {}, min {}".format(len(imgs), len(statistics), len(statistics[soted[0][0]]), len(statistics[soted[-1][0]])))

count = 0
for p in statistics:
    imgs = statistics[p]
    # remove first 2 images
    if len(imgs) > SAMPLES_PER_VIDEO:
        imgs = random.sample(imgs, SAMPLES_PER_VIDEO)

    for img in imgs:
        name = os.path.basename(img)
        dirname = os.path.dirname(img)
        src_img_path = img
        dst_img_path = os.path.join(OUT_IMG_ROOT_DIR, name)

        facerect_name = os.path.splitext(img)[0]+'.facerect'
        src_facerect_path = os.path.join(dirname, facerect_name)
        if not os.path.exists(src_img_path):
            print("Not exist ", src_facerect_path)
        dst_facerect_path = os.path.join(OUT_IMG_ROOT_DIR, facerect_name)

        shutil.copy(src_img_path, dst_img_path)
        shutil.copy(src_facerect_path, dst_facerect_path)
        count += 1

print("Get {} samples!".format(count))