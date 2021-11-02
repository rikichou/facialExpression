import sys
sys.path.append('..')
import os

import cv2

from utils.face_det_python.scrfd import ScrdfFaceDet
from utils import common

IMG_ROOT_DIR = r'E:\workspace\pro\facialExpression\data\shixisheng_selected\youtube\1029\org_rename\wzw'

# model init
fd = ScrdfFaceDet(0.2, model_path='../utils/face_det_python/models/model.pth', config='../utils/face_det_python/models/scrfd_500m.py', device='cuda')

subdirs = os.listdir(IMG_ROOT_DIR)
subdirs.sort()

noface_count = 0
not_img = 0
imgs_count = 0
for subdir in subdirs:
    subdir_path = os.path.join(IMG_ROOT_DIR, subdir)

    imgs = os.listdir(subdir_path)
    imgs_count += len(imgs)
    for idx,img in enumerate(imgs):
        src_img_path = os.path.join(subdir_path, img)

        # check if is image
        if not common.is_image(src_img_path):
            print("{} is not image".format(src_img_path))
            not_img += 1
            continue

        # get face rect
        image = cv2.imread(src_img_path)
        if image is None:
            print("Failed to open ", src_img_path)
            continue
        result = fd.forward(image)
        if len(result)<1:
            noface_count += 1
            continue

        # save rectangle
        out_rect_file_path = os.path.join(subdir_path, img.rsplit('.', maxsplit=1)[0]+'.facerect')
        if os.path.exists(out_rect_file_path):
            print(out_rect_file_path)
        common.save_lists_to_txtfile(result, out_rect_file_path)

        if idx%500 == 0:
            print("{} {}/{}".format(subdir, idx, len(imgs)))

        # for box in result:
        #     rect = box[:4]
        #     prob = box[4]
        #     cv2.rectangle(image, (rect[0],rect[1]), (rect[2],rect[3]), (255,0,0), 1)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
print("total {}, {} have no face! {} not image".format(imgs_count, noface_count, not_img))
