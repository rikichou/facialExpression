import glob
import os.path
import shutil
import sys

src_dir = '/home/ruiming/workspace/pro/facialExpression/data/ExpW/data/image/rectangle/angry'
dst_dir = '/home/ruiming/workspace/pro/facialExpression/data/FacialExpressionOrignialData/ExpW/train/angry'

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

def is_image(name):
    img_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
    prefix,ext = os.path.splitext(name)
    ext = ext.lower()
    if ext in img_ext:
        return True
    else:
        return False

src_imgs_path = os.listdir(src_dir)
count = 0
for idx,img_name in enumerate(src_imgs_path):
    src_img_path = os.path.join(src_dir, img_name)
    if not is_image(img_name):
        continue
    dst_img_path = os.path.join(dst_dir, img_name)
    shutil.copy(src_img_path, dst_img_path)

    if idx%1000 == 0:
        print("{}/{}".format(idx, len(src_imgs_path)))
    count += 1
print("Total {} images".format(count))