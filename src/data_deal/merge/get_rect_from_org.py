import os
import shutil
import glob

ROOT_IMG_DIR = r'E:\workspace\pro\facialExpression\data\select_finish2\RAFDB'
OUT_ROOT_IMG_DIR = r'E:\workspace\pro\facialExpression\data\select_finish2_rename\Selected\rafdb'

org_imgs = glob.glob(ROOT_IMG_DIR + r'\*\*\*')
out_imgs = glob.glob(OUT_ROOT_IMG_DIR + r'\*\*\*')

imgs_to_path = {}
for img in org_imgs:
    imgs_to_path[os.path.basename(img)] = img

for img in out_imgs:
    name = os.path.basename(img)
    try:
        if 'Oulu' in ROOT_IMG_DIR:
            org_name = name.split('_', 4)[4]
        else:
            org_name = name.split('_', 1)[1]
    except:
        print("Error occur ", name)
    dst_label_path = os.path.splitext(img)[0]+'.facerect'
    if os.path.exists(dst_label_path):
        continue

    if org_name in imgs_to_path:
        # get src image and label file path
        org_img_path = imgs_to_path[org_name]
        org_label_path = os.path.join(os.path.dirname(org_img_path), os.path.splitext(org_name)[0]+'.facerect')
        if not os.path.exists(org_label_path):
            print("Warning {} not found".format(org_label_path))
            continue
        shutil.copy(org_label_path, dst_label_path)
    else:
        print("Warning {} not found in org images".format(img))



