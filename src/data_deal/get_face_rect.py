import sys
import os
print(sys.platform)
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
from common_utils.face_det_python import scrfd
from common_utils.scn_python import scn
from common_utils.mmcls_python import mmcls_fer
from common_utils import common

from multiprocessing import Process, Lock, Value

def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument(
        'src_folder', type=str, help='root directory for the frames')
    parser.add_argument(
        'out_dir', type=str, help='out dir name')
    parser.add_argument(
        '--ext', type=str, default='jpg', help='out name')
    parser.add_argument(
        '--use_scrfd',
        action='store_true',
        default=False,
        help='choose face detection handler, yolov5 or scrfd')
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
    if args.use_scrfd:
        # init face detection model
        fd = scrfd.ScrdfFaceDet(0.45,
                                model_path='common_utils/face_det_python/models/model.pth',
                                device='cpu' if args.cpu else 'cuda',
                                config='common_utils/face_det_python/models/scrfd_500m.py')
    else:
        from FaceDetection import FaceDetect
        args.weights = os.path.join(yolov5_src, 'weights/200_last.pt')
        args.imgsz = 640
        faceDetect = FaceDetect(args=args)

    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            continue

        # face detection
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
            bbox = faceDetect.detect(image)
            # print(bbox)
            if len(bbox) != 4 or sum(bbox) < 1:
                continue
            else:
                sx, sy, ex, ey = bbox

        # debug
        cv2.rectangle(image, (sx, sy), (ex, ey), (255, 0, 0), 10)
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
