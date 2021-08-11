import glob
import os.path
import shutil

imgdir = '/home/ruiming/workspace/pro/facialExpression/data/Oulu_CASIA_NIR_VIS/*/*/*/*/*.jpeg'
out_img_root_dir = '/home/ruiming/workspace/pro/facialExpression/data/Oulu_CASIA_NIR_VIS/cat'
if not os.path.exists(out_img_root_dir):
    os.makedirs(out_img_root_dir)
imgs = glob.glob(imgdir)


for img in imgs:
    infos = img.rsplit('/', maxsplit=5)
    cat_name = infos[-2]

    # out image info
    out_img_dir = os.path.join(out_img_root_dir, cat_name)
    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)
    out_img_path = os.path.join(out_img_dir, '_'.join(infos[1:]))
    shutil.copy(img, out_img_path)