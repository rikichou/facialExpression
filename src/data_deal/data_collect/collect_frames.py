import sys
import os

if sys.platform == "darwin":
    print('=>>>>load data from mac platform')
    sys.path.append("/Users/zhourui/workspace/pro/source/yolov5")
    yolov5_src = "/Users/zhourui/workspace/pro/source/yolov5"
elif sys.platform == 'win32':
    print('=>>>>load data from window platform')
    #sys.path.append(r"D:\workspace\pro\source\yolov5")
    sys.path.append(r"E:\workspace\pro\facialExpression\src\data_deal")
    yolov5_src = r"D:\workspace\pro\source\yolov5"
else:
    print('=>>>>load data from linux platform')
    sys.path.append("/zhourui/workspace/pro/facialExpression/src/data_deal")
    yolov5_src = "/home/ruiming/workspace/pro/source/yolov5"

import argparse
import glob
from pathlib import Path
import cv2


from utils.scn_python import scn
from utils.mmcls_python import mmcls_fer
from utils.whenet_fpose_python import whenet_fpose
from utils import common

from multiprocessing import Process, Lock, Value

dir_label_map = {0:'Angry', 1:'Happy', 2:'Neutral', 3:'Sad', 100:'pose', 101:'blur', 102:'size'}
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
        default=True,
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

def get_out_face_region(image, rect):
    sx, sy, ex, ey = rect
    h, w, c = image.shape
    faceh = ey - sy
    facew = ex - sx

    longsize = max(faceh, facew)
    longsize = longsize + longsize*0.4
    expendw = longsize - facew
    expendh = longsize - faceh

    sx = sx - (expendw / 2)
    ex = ex + (expendw / 2)
    sy = sy - (expendh / 2)
    ey = ey + (expendh / 2)

    sx = int(max(0, sx))
    sy = int(max(0, sy))
    ex = int(min(w - 1, ex))
    ey = int(min(h - 1, ey))

    return image[sy:ey, sx:ex, :], sx, sy, ex, ey

def frame_deal(paths, args, lock, counter, total_length):
    # init face detection model
    if args.use_scrfd:
        from utils.face_det_python import scrfd
        fd = scrfd.ScrdfFaceDet(0.45,
                                model_path='utils/face_det_python/models/model.pth',
                                device='cpu' if args.cpu else 'cuda',
                                config='utils/face_det_python/models/scrfd_500m.py')
    else:
        from FaceDetection import FaceDetect
        args.weights = os.path.join(yolov5_src, 'weights/200_last.pt')
        args.imgsz = 640
        fd = FaceDetect(args=args)

    # init facial expression model : mmclassification
    fer_mmcls = mmcls_fer.MMCLSFer(config_file_path='utils/mmcls_python/models/resnet_18_dms_rgbnir/resnet_18_dms_rgbnir.py',
                                   ckpt_path='utils/mmcls_python/models/resnet_18_dms_rgbnir/epoch_16.pth',
                                   device='cpu' if args.cpu else 'cuda')
    # init facial expression model : SCN
    fer_scn = scn.ScnFacialExpressionCat(model_path='utils/scn_python/models/epoch26_acc0.8615.pth', device='cpu' if args.cpu else 'cuda')

    # init face pose
    #fpose = pose.Pose('utils/face_pose_python/model/aver_error_2.2484_epoch_53_multi.pkl', args.cpu)
    # init face pose
    fpose = whenet_fpose.Pose('../data_deal/utils/whenet_fpose_python/model/WHENet.h5')

    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            continue
        image = frame
        bboxes = []
        if args.use_scrfd:
            result = fd.forward(image)
            if len(result) < 1:
                continue
            for box in result:
                sx, sy, ex, ey, prob = box
                if prob < 0.45:
                    continue
                bboxes.append([sx,sy,ex,ey])
        else:
            bbox = fd.detect(image)
            # print(bbox)
            if len(bbox) != 4 or sum(bbox) < 1:
                continue
            else:
                bboxes.append(bbox)
        count = 0
        for bbox in bboxes:
            sx,sy,ex,ey = bbox

            # get face image
            face_image,fsx,fsy,fex,fey = get_out_face_region(image, (sx,sy,ex,ey))

            # Check 1: check if the bbox is too small
            if ey-sy < 60:
                dname = 'size'
                dst_img_path = os.path.join(args.out_dir, dname,
                                            os.path.basename(os.path.dirname(path)) + '_' + os.path.basename(path))
                cv2.imwrite(dst_img_path, face_image)
                continue

            # Check 2: check if the angle is out of range
            yaw_max = 45
            pitch_max = 30
            yaw, pitch, roll = fpose(image, (sx, sy, ex, ey))
            if abs(yaw)>=yaw_max or abs(pitch)>=pitch_max:
                dname = 'pose'
                dst_img_path = os.path.join(args.out_dir, dname,
                                            os.path.basename(os.path.dirname(path)) + '_' + os.path.basename(path))
                cv2.imwrite(dst_img_path, face_image)
                continue

            # facial expression
            pred_label, pred_sclore, pred_name = fer_mmcls(image, [sx,sy,ex,ey])
            #pred_label, pred_sclore, pred_name = fer_scn(image, [sx, sy, ex, ey])

            # copy image to sub dir
            count += 1
            dname = dir_label_map[pred_label]
            dst_img_path = os.path.join(args.out_dir, dname, os.path.basename(os.path.dirname(path))+'_'+os.path.basename(path))
            cv2.imwrite(dst_img_path, face_image)
            #print(dst_img_path)

            # debug
            # cv2.rectangle(image, (fsx, fsy), (fex, fey), (255, 0, 0), 10)
            # cv2.putText(image, '{}:{:.3f}'.format(pred_name, pred_sclore), (sx, sy - 20),
            #             0, 2, (0, 0, 255), 2)
            # cv2.imshow('debug', image)
            # if cv2.waitKey(0) & 0xff == ord('q'):
            #     break

        # counter
        lock.acquire()
        try:
            # p_bar.update(1)
            counter.value += 1
            if counter.value % 50 == 0:
                print(f"{counter.value}/{total_length} done.")
        finally:
            lock.release()
    # debug
    cv2.destroyAllWindows()

def multi_process(frames_list, args):
    process_num = args.num_worker
    # check if is valid
    to_process_frames_list = [x for x in frames_list if common.is_image(x)]

    # start process
    files = to_process_frames_list
    grid_size = len(files) // process_num
    process_pool = []
    lock = Lock()
    counter = Value("i", 0)
    for i in range(process_num):
        start_index = grid_size * i
        if i != process_num - 1:
            end_index = grid_size * (i + 1)
        else:
            end_index = len(files)
        pw = Process(target=frame_deal,
                     args=(files[start_index:end_index], args, lock, counter, len(files)))
        pw.start()
        process_pool.append(pw)

    for p in process_pool:
        p.join()

def main():
    args = parse_args()

    # create out dir
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    for k in dir_label_map:
        dname = dir_label_map[k]
        dpath = os.path.join(args.out_dir, dname)
        if not os.path.isdir(dpath):
            os.makedirs(dpath)

    # get all images folders
    frames_list = glob.glob(os.path.join(args.src_folder, str(Path('*/'*args.level))))

    # multi process
    multi_process(frames_list, args)

if __name__ == '__main__':
    main()

