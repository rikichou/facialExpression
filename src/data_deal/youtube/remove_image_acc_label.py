import glob
import os
import shutil

IMG_ROOT_DIR = r'E:\workspace\pro\facialExpression\data\select_finish2_rename\Selected\train\neutral'
files = glob.glob(IMG_ROOT_DIR+'\*')

OUT_IMG_ROOT_DIR = r'E:\workspace\pro\facialExpression\data\select_finish2_rename\Selected\train\neutral_out'
if not os.path.exists(OUT_IMG_ROOT_DIR):
    os.makedirs(OUT_IMG_ROOT_DIR)

def is_image(name):
    img_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
    prefix,ext = os.path.splitext(name)
    ext = ext.lower()
    if ext in img_ext:
        return True
    else:
        return False

count = 0
count_img = 0
for idx,f in enumerate(files):
    # check if is image
    fname = os.path.basename(f)
    if not is_image(fname):
        continue

    # get facerect
    facerect_name = os.path.splitext(fname)[0]+'.facerect'
    facerect_file_path = os.path.join(IMG_ROOT_DIR, facerect_name)
    if not os.path.exists(facerect_file_path):
        count += 1
        continue

    # copy
    new_img_path = os.path.join(OUT_IMG_ROOT_DIR, fname)
    new_facerect_path = os.path.join(OUT_IMG_ROOT_DIR, facerect_name)

    shutil.copy(f, new_img_path)
    shutil.copy(facerect_file_path, new_facerect_path)

    count_img += 1

print("Get {} images".format(count_img))
print("{} have no facerect!".format(count))