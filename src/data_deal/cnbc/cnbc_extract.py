import glob
import os.path
import shutil

imgdir = '/home/ruiming/workspace/pro/facialExpression/data/CNBC/*/*/*.jpg'
out_img_root_dir = '/home/ruiming/workspace/pro/facialExpression/data/CNBC/cat'
if not os.path.exists(out_img_root_dir):
    os.makedirs(out_img_root_dir)

label_map = {'SA':'sad', 'HA':'happy', 'FE':'fear', 'DI':'disgust',
             'AN':'anger', 'CO':'confusion', 'NE':'neutral', 'SU':'surprise'}

imgs = glob.glob(imgdir)

statistics_label = {}
for img in imgs:
    imgname = os.path.basename(img)
    imgprefix,_ = os.path.splitext(imgname)

    # check if has expression label
    cat_name = imgprefix.split('_')[-1]
    if cat_name not in label_map:
        continue
    cat_name = label_map[cat_name]

    # copy image
    src_img_path = img
    out_img_dir = os.path.join(out_img_root_dir, cat_name)
    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)
    out_img_path = os.path.join(out_img_dir, imgname)
    if os.path.exists(out_img_path):
        print("!!!!!!!!!!!!!!! Warning")
        continue
    shutil.copy(src_img_path, out_img_path)

    # statistics
    if cat_name not in statistics_label:
        statistics_label[cat_name] = 0
    statistics_label[cat_name] += 1

print(statistics_label)