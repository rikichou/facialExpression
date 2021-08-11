import sys
import os

img_root_dir = '/home/ruiming/workspace/pro/facialExpression/data/out/FacialExpressionOrignialData/AffectNet'
train_root_dir = os.path.join(img_root_dir, 'val')
out_label_txt_path = os.path.join(img_root_dir, 'affectnet_val.txt')

exp_cats = {'Angry':3, 'Sad':2, 'Happy':1, 'Neutral':0}

def is_image(name):
    img_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
    prefix,ext = os.path.splitext(name)
    ext = ext.lower()
    if ext in img_ext:
        return True
    else:
        return False

with open(out_label_txt_path, 'w') as fp:
    for cat in exp_cats:
        cat_dir = os.path.join(train_root_dir, cat)
        imgs = os.listdir(cat_dir)
        for img in imgs:
            if not is_image(img):
                continue
            fp.write('{} {}\n'.format(os.path.join('val', cat, img), exp_cats[cat]))


