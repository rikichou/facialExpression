import sys
import os
print(sys.platform)
if sys.platform == "darwin":
    print('=>>>>load data from mac platform')
    sys.path.append('/Users/zhourui/workspace/pro/facialExpression/src/data_deal')
elif sys.platform == 'win32':
    print('=>>>>load data from window platform')
    sys.path.append(r"D:\workspace\pro\facialExpression\src\data_deal")
else:
    print('=>>>>load data from linux platform')
    sys.path.append("~/workspace/pro/facialExpression/src/data_deal")

import argparse
import glob
from pathlib import Path
import cv2

from utils.face_det_python import scrfd
from utils.scn_python import scn
from utils.mmcls_python import mmcls_fer
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

def frame_deal(paths, args, lock, counter, total_length):
    # init face detection model
    fd = scrfd.ScrdfFaceDet(0.45,
                            model_path='../utils/face_det_python/models/model.pth',
                            device='cpu' if args.cpu else 'cuda',
                            config='../utils/face_det_python/models/scrfd_500m.py')

    # init facial expression model : mmclassification
    fer_mmcls = mmcls_fer.MMCLSFer(config_file_path=os.path.join(args.mmcls_src_dir, 'configs/fer/mobilenet_v2.py'),
                                   ckpt_path=os.path.join(args.mmcls_src_dir, 'run/fer/mobilenetv2/latest.pth'),
                                   device='cpu' if args.cpu else 'cuda')
    # init facial expression model : SCN
    fer_scn = scn.ScnFacialExpressionCat(model_path='../utils/scn_python/models/epoch26_acc0.8615.pth', device='cpu' if args.cpu else 'cuda')

    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            continue

        # face detection
        image = frame
        result = fd.forward(image)
        if len(result) < 1:
            continue
        box = result[0]
        sx, sy, ex, ey, prob = box
        if prob < 0.45:
            continue

        # facial expression
        #pred_label, pred_sclore, pred_name = fer_mmcls(image, [sx,sy,ex,ey])
        pred_label, pred_sclore, pred_name = fer_scn(image, [sx, sy, ex, ey])

        # debug
        cv2.rectangle(image, (sx, sy), (ex, ey), (255, 0, 0), 10)
        cv2.putText(image, '{}:{:.3f}'.format(pred_name, pred_sclore), (sx, sy - 20),
                    0, 2, (0, 0, 255), 2)
        cv2.imshow('debug', image)
        if cv2.waitKey(0) & 0xff == ord('q'):
            break

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

    # get all images folders
    frames_list = glob.glob(os.path.join(args.src_folder, str(Path('*/'*args.level))))

    # multi process
    multi_process(frames_list, args)

if __name__ == '__main__':
    main()

