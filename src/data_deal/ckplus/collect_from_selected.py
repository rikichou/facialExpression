import os
import shutil
import glob

IMG_ROOT_DIR = r'E:\workspace\pro\facialExpression\data\shixisheng_selected\ckplus'
OUT_IMG_ROOT_DIR = r'E:\workspace\pro\facialExpression\data\shixisheng_selected\ckplus_cat'

imgs = glob.glob(IMG_ROOT_DIR + r'\*\*\*')

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
    dataset,person,cat,name = img.rsplit('\\', maxsplit=4)[1:]
    if not is_image(name):
        print("Skip ", name)
        continue

    new_name = '_'.join([dataset,name])
    out_dir = os.path.join(OUT_IMG_ROOT_DIR, 'train', cat)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, new_name)

    shutil.copy(img, out_path)