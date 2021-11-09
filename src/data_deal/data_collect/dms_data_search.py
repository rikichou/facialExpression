import sys
import os
print(sys.platform)
if sys.platform == "darwin":
    print('=>>>>load data from mac platform')
    sys.path.append("/Users/zhourui/workspace/pro/source/yolov5")
elif sys.platform == 'win32':
    print('=>>>>load data from window platform')
    sys.path.append(r"D:\workspace\pro\source\yolov5")
else:
    print('=>>>>load data from linux platform')
    sys.path.append(r"/zhourui/workspace/pro/source/yolov5")

import sys
sys.path.append(os.path.join(os.getcwd(), '../../common_utils'))
import glob
import shutil
import argparse
import multiprocessing
from multiprocessing import Process, Lock, Value
import cv2
import numpy as np
from pathlib import Path

from mmcls_python import mmcls_fer
from FaceDetection import FaceDetect

def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument(
        'src_folder', type=str, help='root directory for the frames')
    parser.add_argument(
        'out_folder', type=str, help='save name')
    parser.add_argument(
        '--level',
        type=int,
        default=3,
        choices=[1, 2, 3],
        help='directory level of data')
    parser.add_argument(
        '--num_worker',
        type=int,
        default=10,
        help='num workers to preprocess')
    parser.add_argument('--weights',
                        default=r'/zhourui/workspace/pro/source/yolov5/weights/200_last.pt',
                        help='experiment', type=str)
    parser.add_argument('--imgsz',
                        default=640,
                        help='experiment', type=int)
    parser.add_argument(
        '--cpu',
        action='store_true',
        default=False,
        help='whether to use cpu')

    args = parser.parse_args()
    return args

def process_paths(paths, args, lock, counter, total_length):
    # init face detection model
    faceDetect = FaceDetect(args=args)

    # facial expression
    model_name = 'res14_clean1029_5cls'
    epoch = 100
    input_channels = 3

    fer_mmcls = mmcls_fer.MMCLSFer(config_file_path='../data_deal/common_utils/mmcls_python/models/{}/{}.py'.format(model_name, model_name),
                                   ckpt_path='../data_deal/common_utils/mmcls_python/models/{}/epoch_{}.pth'.format(model_name, epoch),
                                   device='cpu' if args.cpu else 'cuda',
                                   input_channels=input_channels)

    for path in paths:
        # get all images to process
        images_path = glob.glob(os.path.join(path, '*.jpg'))

        for i in range(0, len(images_path), 20):
            img_path = images_path[i]

            # check if run before
            _, dname, imgname = img_path.rsplit('/', maxsplit=2)

            # open and preprocess image
            image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
            if image is None:
                print("!!!!!!!!!!!!!!!!!!!! Failed to read image ", img_path)
                assert (False)

            # get face rectangle, have no face rectangle, set None
            bbox = faceDetect.detect(image)
            # print(bbox)
            if len(bbox) != 4 or sum(bbox) < 1:
                continue

            sx, sy, ex, ey = bbox

            # get smoke keypoint
            pred_label, pred_sclore, pred_name = fer_mmcls(image, [sx, sy, ex, ey])

            if pred_label == 2:
                continue
            else:
                out_dname = pred_name
                out_dir = os.path.join(args.out_folder, out_dname)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                out_v_dir = os.path.join(out_dir, dname)
                if not os.path.exists(out_v_dir):
                    os.makedirs(out_v_dir)

                out_img_path = os.path.join(out_v_dir, '{:.3f}_'.format(pred_sclore) + imgname)
                if not os.path.exists(out_img_path):
                    shutil.copy(img_path, out_img_path)

        # counter
        lock.acquire()
        try:
            # p_bar.update(1)
            counter.value += 1
            if counter.value % 50 == 0:
                print(f"{counter.value}/{total_length} done.")
        finally:
            lock.release()


def multi_process(video_dirs, args):
    process_num = args.num_worker
    # check if is valid
    to_process_video_dirs = []
    for v in video_dirs:
        if not os.path.isdir(v):
            continue
        to_process_video_dirs.append(v)

    # start process
    files = to_process_video_dirs
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
        pw = Process(target=process_paths,
                     args=(files[start_index:end_index], args, lock, counter, len(files)))
        pw.start()
        process_pool.append(pw)

    for p in process_pool:
        p.join()

def main():
    args = parse_args()

    # get all video folders
    video_dirs = glob.glob(os.path.join(args.src_folder, str(Path('*/'*args.level))))

    # check if dir is dealed
    print("Found {} videos! {} videos not yet processed!".format(len(video_dirs), len(video_dirs)))

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    # multi process
    multi_process(video_dirs, args)

multiprocessing.set_start_method('forkserver', force=True)
if __name__ == '__main__':
    main()