import sys
import os
import glob

IMG_ROOT_DIR = r'E:\software\nvtai_tool\config\facial_expression\cnn25\data\images_gray'
images_list_file_path = r'E:\software\nvtai_tool\config\facial_expression\cnn25\data\images_gray\image_list.txt'

imgs = glob.glob(IMG_ROOT_DIR + r'\*\*')

def is_image(name):
    img_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
    prefix,ext = os.path.splitext(name)
    ext = ext.lower()
    if ext in img_ext:
        return True
    else:
        return False

with open(images_list_file_path,'w') as fp:
    for idx,img in enumerate(imgs):
        if not is_image(img):
            continue
        _, dname, imgname = img.rsplit('\\', maxsplit=2)
        fp.write('\\'.join([dname, imgname])+'\n')
