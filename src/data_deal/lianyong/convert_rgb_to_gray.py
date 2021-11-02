import os
import cv2
import glob

IMG_ROOT_DIR = r'E:\software\nvtai_tool\config\facial_expression\cnn25\data\images'
OUT_IMG_ROOT_DIR = r'E:\software\nvtai_tool\config\facial_expression\cnn25\data\images_gray'

if not os.path.exists(OUT_IMG_ROOT_DIR):
    os.makedirs(OUT_IMG_ROOT_DIR)

imgs = glob.glob(IMG_ROOT_DIR + r'\*\*')

def is_image(name):
    img_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
    prefix,ext = os.path.splitext(name)
    ext = ext.lower()
    if ext in img_ext:
        return True
    else:
        return False

for idx,img in enumerate(imgs):
    if not is_image(img):
        continue
    image_bgr = cv2.imread(img)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _,dname,imgname = img.rsplit('\\', maxsplit=2)
    out_dir = os.path.join(OUT_IMG_ROOT_DIR, dname)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_img_path = os.path.join(out_dir, imgname)
    cv2.imwrite(out_img_path, image_gray)

    if idx%100 == 0:
        print("{}/{}".format(idx,len(imgs)))
