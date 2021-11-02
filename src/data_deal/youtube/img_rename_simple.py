import sys
import os
import glob
import shutil

IMG_ROOT_DIR = r'G:\pro\facialexpression\data\youtube\expression\N'

imgs = glob.glob(IMG_ROOT_DIR + r'\*')

def is_image(name):
    img_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
    prefix,ext = os.path.splitext(name)
    ext = ext.lower()
    if ext in img_ext:
        return True
    else:
        return False

for img in imgs:
    # get old info
    basedir = os.path.dirname(img)
    basename = os.path.basename(img)

    new_name = 'youtube_'+basename
    new_path = os.path.join(basedir, new_name)

    shutil.copy(img, new_path)
    os.remove(img)