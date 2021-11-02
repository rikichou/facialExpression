import cv2
import os
from utils import common
import glob

img_dir = r'E:\workspace\pro\facialExpression\data\shixisheng_selected\youtube\1026_occ\org_rename\riki\neutral_bigangle'
imgs = glob.glob(os.path.join(img_dir, '*.jpg'))

for img_path in imgs:
    facerect_path = img_path[:-3]+'facerect'

    infos = common.read_lists_from_txtfile(facerect_path)

    image = cv2.imread(img_path)

    for info in infos:
        xmin,ymin,xmax,ymax = info[:4]
        image = cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,0,255), 2)

    cv2.imshow('1', image)
    key = cv2.waitKey(0)
    if key==ord('q'):
        break