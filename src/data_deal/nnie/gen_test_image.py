import os
import cv2
import numpy as np

def gen_bgr_image(src_img_path, dst_bgr_path, bgr_shape):
    img = cv2.imread(src_img_path)
    img = cv2.resize(img, (bgr_shape, bgr_shape))
    img = np.transpose(img, (2,0,1))
    img = img.flatten()

    img.tofile(dst_bgr_path)

def show_gbr_image(bgr_img_path, bgr_shape):
    a = np.fromfile(bgr_img_path, np.uint8)
    b = np.reshape(a, (3, bgr_shape, bgr_shape))
    c = np.transpose(b, (1, 2, 0))

    cv2.imshow('1', c)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

dst_size = 112

# gen from image path
#img_path = r'E:\software\haisi\RuyiStudio-2.0.46\workspace\facialExpression\images\1d51a696cb8de03eea54cba689635227693d6a1e8f0f1a1ce7588ca1.jpg'
#out_img_path = r'E:\software\haisi\RuyiStudio-2.0.46\workspace\facialExpression\images\62801fd29f5627c592f7a81f7d431af9c50fa407af16beddb419f5a1.bgr'
#gen_bgr_image(img_path, out_img_path, dst_size)

# gen from image file
img_list_file = r'E:\software\haisi\Hi3516C V500R001C02SPC020\SVP_PC\HiSVP_PC_V1.2.2.2\software\data\fer\images\imglist.txt'
with open(img_list_file, 'r') as fp:
    for f in fp.readlines():
        img_path = f.strip()
        out_img_path = os.path.splitext(f)[0] + '.bgr'
        gen_bgr_image(img_path, out_img_path, dst_size)

#show_gbr_image(r'E:\software\haisi\RuyiStudio-2.0.46\workspace\facialExpression\images\62801fd29f5627c592f7a81f7d431af9c50fa407af16beddb419f5a1.bgr', 112)