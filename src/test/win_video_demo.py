import sys
import os

if sys.platform == "darwin":
    print('=>>>>load data from mac platform')
    sys.path.append("/Users/zhourui/workspace/pro/source/yolov5")
    yolov5_src = "/Users/zhourui/workspace/pro/source/yolov5"
elif sys.platform == 'win32':
    print('=>>>>load data from window platform')
    #sys.path.append(r"D:\workspace\pro\source\yolov5")
    yolov5_src = r"D:\workspace\pro\source\yolov5"
else:
    print('=>>>>load data from linux platform')
    sys.path.append("/home/ruiming/workspace/pro/source/yolov5")
    yolov5_src = "/home/ruiming/workspace/pro/source/yolov5"

import argparse
import glob
from pathlib import Path
import cv2

from utils.face_det_python import scrfd
from utils.scn_python import scn
from utils.mmcls_python import mmcls_fer
from utils.face_pose_python import pose
from utils.whenet_fpose_python import whenet_fpose
from utils import common

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

def test_video(video_path, args):
    args.use_scrfd = True
    args.cpu = False

    # init face detection model
    if args.use_scrfd:
        fd = scrfd.ScrdfFaceDet(0.45,
                                model_path='../data_deal/utils/face_det_python/models/model.pth',
                                device='cpu' if args.cpu else 'cuda',
                                config='../data_deal/utils/face_det_python/models/scrfd_500m.py')
    else:
        from FaceDetection import FaceDetect
        args.weights = os.path.join(yolov5_src, 'weights/200_last.pt')
        args.imgsz = 640
        fd = FaceDetect(args=args)

    # init facial expression model : mmclassification
    fer_mmcls = mmcls_fer.MMCLSFer(config_file_path='../data_deal/utils/mmcls_python/models/resnet_18_dms_rgbnir/resnet_18_dms_rgbnir.py',
                                   ckpt_path='../data_deal/utils/mmcls_python/models/resnet_18_dms_rgbnir/epoch_16.pth',
                                   device='cpu' if args.cpu else 'cuda')
    # init facial expression model : SCN
    fer_scn = scn.ScnFacialExpressionCat(model_path='../data_deal/utils/scn_python/models/epoch26_acc0.8615.pth', device='cpu' if args.cpu else 'cuda')

    # init face pose
    #fpose = pose.Pose('../data_deal/utils/face_pose_python/model/aver_error_2.2484_epoch_53_multi.pkl', args.cpu)
    fpose = whenet_fpose.Pose('../data_deal/utils/whenet_fpose_python/model/WHENet.h5')

    if True:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        #frame = cv2.imread(r'E:\workspace\pro\facialExpression\data\test\angry1.png')
        frame = cv2.resize(frame,(704,396))

        image = frame.copy()
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

        # face pose
        # yaw,pitch,roll = fpose(image, (sx,sy,ex,ey))
        # print(yaw)
        # print(pitch)
        # print(roll)

        # facial expression
        #pred_label, pred_sclore, pred_name = fer_mmcls(frame, [0, 0, 207, 288])
        #pred_label, pred_sclore, pred_name = fer_mmcls(image, [sx,sy,ex,ey])
        pred_label, pred_sclore, pred_name = fer_scn(image, [sx, sy, ex, ey])
        print(pred_label, pred_sclore, pred_name)
        # debug
        cv2.rectangle(image, (sx, sy), (ex, ey), (255, 0, 0), 10)
        # cv2.putText(image, '{:.2f} {:.2f} {:.2f}'.format(yaw, pitch, roll), (100, 50),
        #             0, 1, (0, 0, 255), 1)
        cv2.putText(image, '{}:{:.3f}'.format(pred_name, pred_sclore), (sx, sy - 5),
                    0, 1, (0, 0, 255), 1)
        cv2.imshow('debug', image)
        #cv2.imwrite(r'E:\workspace\pro\facialExpression\data\test\angry1_out.jpg', image)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    # debug
    cv2.destroyAllWindows()

VIDEO_PATH = r'E:\workspace\pro\facialExpression\data\test\test.mp4'

class Obj(object):
    def __init__(self):
        super().__init__()

args = Obj()

test_video(VIDEO_PATH, args)

