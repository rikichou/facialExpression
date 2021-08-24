import sys
import os

if sys.platform == "darwin":
    print('=>>>>load data from mac platform')
    sys.path.append("/Users/zhourui/workspace/pro/source/yolov5")
    yolov5_src = "/Users/zhourui/workspace/pro/source/yolov5"
elif sys.platform == 'win32':
    print('=>>>>load data from window platform')
    sys.path.append(r"D:\workspace\pro\source\yolov5")
    yolov5_src = r"D:\workspace\pro\source\yolov5"
else:
    print('=>>>>load data from linux platform')
    sys.path.append("/home/ruiming/workspace/pro/source/yolov5")
    yolov5_src = "/home/ruiming/workspace/pro/source/yolov5"

import argparse
import glob
from pathlib import Path
import cv2

from common_utils.face_det_python import scrfd
from common_utils.scn_python import scn
from common_utils.mmcls_python import mmcls_fer
from common_utils import common

from multiprocessing import Process, Lock, Value

dir_label_map = {0:'Angry', 1:'Happy', 2:'Neutral', 3:'Sad'}
#dir_label_map = {'angry_heavy':0, 'angry_light':0, 'happy':1, 'neutral':2, 'sad_heavy':3, 'sad_light':3}
#dir_label_map = {'angry_heavy': 0, 'happy': 1, 'neutral': 2, 'sad_heavy': 3}

def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument(
        'src_folder', type=str, help='root directory for the frames')
    parser.add_argument(
        'out_dir', type=str, help='out name')
    parser.add_argument(
        '--use_scrfd',
        action='store_true',
        default=False,
        help='choose face detection handler, yolov5 or scrfd')
    parser.add_argument(
        '--ext', type=str, default='jpg', help='out name')
    parser.add_argument(
        '--mmcls_src_dir', default='/Users/zhourui/workspace/pro/source/mmclassification', type=str, help='mmclassification source dir')
    parser.add_argument(
        '--level',
        type=int,
        default=1,
        choices=[1, 2, 3],
        help='directory level of data')
    parser.add_argument(
        '--num_worker',
        type=int,
        default=10,
        help='num workers to preprocess')
    parser.add_argument(
        '--cpu',
        action='store_true',
        default=False,
        help='whether to use cpu')

    args = parser.parse_args()
    return args

def test_imgs(img_dir, paths, args):
    args.use_scrfd = True
    args.cpu = True

    # init face detection model
    if args.use_scrfd:
        fd = scrfd.ScrdfFaceDet(0.45,
                                model_path='../data_deal/common_utils/face_det_python/models/model.pth',
                                device='cpu' if args.cpu else 'cuda',
                                config='../data_deal/common_utils/face_det_python/models/scrfd_500m.py')
    else:
        from FaceDetection import FaceDetect
        args.weights = os.path.join(yolov5_src, 'weights/200_last.pt')
        args.imgsz = 640
        fd = FaceDetect(args=args)

    # init facial expression model : mmclassification
    fer_mmcls = mmcls_fer.MMCLSFer(config_file_path='../data_deal/common_utils/mmcls_python/models/mobilenet_v2/mobilenet_v2.py',
                                   ckpt_path='../data_deal/common_utils/mmcls_python/models/mobilenet_v2/latest.pth',
                                   device='cpu' if args.cpu else 'cuda')
    # init facial expression model : SCN
    fer_scn = scn.ScnFacialExpressionCat(model_path='../data_deal/common_utils/scn_python/models/epoch26_acc0.8615.pth', device='cpu' if args.cpu else 'cuda')

    for path in paths:
        path = os.path.join(IMG_DIR, path)
        frame = cv2.imread(path)
        if frame is None:
            print("Failed to read image ", path)
            continue
        image = frame
        if args.use_scrfd:
            result = fd.forward(image)
            if len(result) < 1:
                continue
            box = result[0]
            sx, sy, ex, ey, prob = box
            if prob < 0.45:
                continue
        else:
            bbox = fd.detect(image)
            # print(bbox)
            if len(bbox) != 4 or sum(bbox) < 1:
                print("cat not detect face")
                continue
            else:
                sx, sy, ex, ey = bbox

        # facial expression
        pred_label, pred_sclore, pred_name = fer_mmcls(image, [sx,sy,ex,ey])
        #pred_label, pred_sclore, pred_name = fer_scn(image, [sx, sy, ex, ey])
        print(pred_label, pred_sclore, pred_name)
        # debug
        cv2.rectangle(image, (sx, sy), (ex, ey), (255, 0, 0), 10)
        cv2.putText(image, '{}:{:.3f}'.format(pred_name, pred_sclore), (sx, sy - 20),
                    0, 2, (0, 0, 255), 2)
        cv2.imshow('debug', image)
        if cv2.waitKey(0) & 0xff == ord('q'):
            break
    # debug
    cv2.destroyAllWindows()

IMG_DIR = '/Users/zhourui/workspace/pro/tmp/1'
imgs = os.listdir(IMG_DIR)

class Obj(object):
    def __init__(self):
        super().__init__()

args = Obj()

test_imgs(IMG_DIR, imgs, args)

