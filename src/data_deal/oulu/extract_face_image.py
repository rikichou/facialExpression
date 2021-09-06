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
    sys.path.append(r"/media/ruiming/data/workspace/pro/source/yolov5")

import glob
import argparse
from multiprocessing import Process, Lock, Value
import cv2
import numpy as np
from pathlib import Path

from FaceDetection import FaceDetect

def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument(
        'src_folder', type=str, help='root directory for the frames')
    parser.add_argument(
        'out_dir', type=str, help='save name')
    parser.add_argument(
        '--level',
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help='directory level of data')
    parser.add_argument(
        '--ext',
        type=str,
        default='jpeg',
        choices=['jpeg', 'jpg', 'bmp'],
        help='video file extensions')
    parser.add_argument(
        '--num_worker',
        type=int,
        default=10,
        help='num workers to preprocess')
    parser.add_argument('--weights',
                        default=r'D:\workspace\pro\source\yolov5\weights\200_last.pt',
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

def get_input_face(image, rect):
    sx, sy, ex, ey = rect
    h, w, c = image.shape
    faceh = ey - sy
    facew = ex - sx

    longsize = max(faceh, facew)
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

def  process_paths(paths, args, lock, counter, total_length):
    # init face detection model
    faceDetect = FaceDetect(args=args)

    for img_path in paths:
        # open and preprocess image
        #image = cv2.imread(img_path) # there is some problem in Chinese char
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)

        if image is None:
            print("!!!!!!!!!!!!!!!!!!!! Failed to read image ", img_path)
            assert (False)

        # get face rectangle, have no face rectangle, set None
        bbox = faceDetect.detect(image)
        # print(bbox)
        if len(bbox) != 4 or sum(bbox) < 1:
            print("Can not detect any face in ", img_path)
            continue
        else:
            # get and save face image
            out_dir = args.out_dir
            if args.level>1:
                out_dir = os.path.join(args.out_dir, str(Path('/')).join(img_path.split(str(Path('/')))[-1*args.level:-1]))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            out_img_path = os.path.join(out_dir, os.path.basename(img_path))

            sx, sy, ex, ey = bbox
            face_img = get_input_face(image, bbox)

            #cv2.imwrite(out_img_path, face_img)
            # debug
            cv2.rectangle(image, (int(sx), int(sy)), (int(ex), int(ey)), (0,0,255), 1)
            cv2.imshow('1', image)
            cv2.waitKey(0)

    # counter
    lock.acquire()
    try:
        # p_bar.update(1)
        counter.value += 1
        if counter.value % 50 == 0:
            print(f"{counter.value}/{total_length} done.")
    finally:
        lock.release()


def multi_process(fullpath_list, args):
    process_num = args.num_worker

    # start process
    files = fullpath_list
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
    fullpath_list = glob.glob(args.src_folder + str(Path('/*' * args.level)) + '.' +
                              args.ext)

    # multi process
    multi_process(fullpath_list, args)

if __name__ == '__main__':
    main()