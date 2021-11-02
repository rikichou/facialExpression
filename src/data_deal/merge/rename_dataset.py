import os
import shutil
import sys
import glob

ROOT_DIR = r'G:\pro\facialexpression\data\youtube\expression\selected'
OUT_ROOT_DIR = r'G:\pro\facialexpression\data\youtube\expression\selected_rename'

imgs = glob.glob(ROOT_DIR + r'\*\*')

def is_image(name):
    img_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
    prefix,ext = os.path.splitext(name)
    ext = ext.lower()
    if ext in img_ext:
        return True
    else:
        return False

for idx,img in enumerate(imgs):

    cat,name = img.rsplit('\\', maxsplit=2)[1:]
    if not is_image(name):
        continue

    new_name = '_'.join(['youtube',name])
    out_dir = os.path.join(OUT_ROOT_DIR, cat)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, new_name)

    shutil.copy(img, out_path)

    if idx%1000 == 0:
        print("{}/{}".format(idx,len(imgs)))

