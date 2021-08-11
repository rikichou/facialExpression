import os
import shutil

import cv2
import xml.dom.minidom

IMG_ROOT_DIR = '/home/ruiming/workspace/pro/facialExpression/data/org/ExpW/data/image/cat'
IMG_OUT_ROOT_DIR = '/home/ruiming/workspace/pro/facialExpression/data/out/FacialExpressionOrignialData/ExpW/train'
if not os.path.exists(IMG_OUT_ROOT_DIR):
    os.makedirs(IMG_OUT_ROOT_DIR)

def is_image(name):
    img_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
    prefix,ext = os.path.splitext(name)
    ext = ext.lower()
    if ext in img_ext:
        return True
    else:
        return False

def get_xml_infos(xmlpath):
    dom = xml.dom.minidom.parse(xmlpath)
    root = dom.documentElement

    objs = root.getElementsByTagName('object')
    infos = []
    for obj in objs:
        bndbox = obj.getElementsByTagName('bndbox')[0]
        node = bndbox.getElementsByTagName('xmin')[0]
        xmin = int(float(node.childNodes[0].data))
        node = bndbox.getElementsByTagName('ymin')[0]
        ymin = int(float(node.childNodes[0].data))
        node = bndbox.getElementsByTagName('xmax')[0]
        xmax = int(float(node.childNodes[0].data))
        node = bndbox.getElementsByTagName('ymax')[0]
        ymax = int(float(node.childNodes[0].data))
        node = obj.getElementsByTagName('expression')[0]
        expression = node.childNodes[0].data
        infos.append([xmin,ymin,xmax,ymax,expression])

    return infos

category_dirs = os.listdir(IMG_ROOT_DIR)

for cat_name in category_dirs:
    cat_dir = os.path.join(IMG_ROOT_DIR, cat_name)
    if not os.path.isdir(cat_dir):
        continue
    out_cat_dir = os.path.join(IMG_OUT_ROOT_DIR, cat_name)
    if not os.path.exists(out_cat_dir):
        os.makedirs(out_cat_dir)

    files_name = os.listdir(cat_dir)
    for file_name in files_name:
        if not is_image(file_name):
            continue
        img_basename = os.path.splitext(file_name)[0]
        img_path = os.path.join(cat_dir, file_name)

        xml_path = os.path.join(cat_dir, img_basename + '.xml')
        xml_infos = get_xml_infos(xml_path)

        image = cv2.imread(img_path)
        xml_info = xml_infos[0]
        sx, sy, ex, ey, expression = xml_info

        # get expand rectangle
        orgh,orgw,_ = image.shape
        expand_ratio = 0.25
        w = ex-sx+1
        h = ey-sy+1
        expw = w*expand_ratio
        exph = h*expand_ratio
        sx -= expw
        ex += expw
        sy -= exph
        ey += exph

        sx = int(max(0, sx))
        sy = int(max(0, sy))
        ex = int(min(ex, orgw-1))
        ey = int(min(ey, orgh-1))

        image = image[sy:ey, sx:ex, :]
        img_out_path = os.path.join(out_cat_dir, file_name)
        cv2.imwrite(img_out_path, image)

    print('{} complete!'.format(cat_name))

