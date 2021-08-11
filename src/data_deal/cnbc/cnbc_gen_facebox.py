import sys
sys.path.append('..')
import os

import cv2

from utils.face_det_python.scrfd import ScrdfFaceDet
from utils import common

IMG_ROOT_DIR = '/train_tmp/fer/select/finish2/CNBC/train'

# model init
fd = ScrdfFaceDet(0.45, model_path='../utils/face_det_python/models/model.pth', config='../utils/face_det_python/models/scrfd_500m.py', device='cuda')

subdirs = os.listdir(IMG_ROOT_DIR)
subdirs.sort()

noface_count = 0
for subdir in subdirs:
    subdir_path = os.path.join(IMG_ROOT_DIR, subdir)
    if not os.path.isdir(subdir_path):
        continue

    imgs = os.listdir(subdir_path)
    for idx,img in enumerate(imgs):
        src_img_path = os.path.join(subdir_path, img)

        # check if is image
        if not common.is_image(src_img_path):
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
        out_rect_file_path = os.path.join(subdir_path, img.rsplit('.')[0]+'.facerect')
        common.save_lists_to_txtfile(result, out_rect_file_path)

        if idx%500 == 0:
            print("{} {}/{}".format(subdir, idx, len(imgs)))

        # for box in result:
        #     rect = box[:4]
        #     prob = box[4]
        #     cv2.rectangle(image, (rect[0],rect[1]), (rect[2],rect[3]), (255,0,0), 1)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
print("{} have no face!".format(noface_count))
