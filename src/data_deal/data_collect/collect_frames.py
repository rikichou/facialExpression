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
from utils import common
from mmcls.apis import inference_model, init_model, show_result_pyplot
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

def get_input_face(image, rect):
    sx,sy,ex,ey = rect
    h,w,c = image.shape
    faceh = ey-sy
    facew = ex-sx

    longsize = max(faceh, facew)
    expendw = longsize-facew
    expendh  = longsize-faceh

    sx = sx-(expendw/2)
    ex = ex+(expendw/2)
    sy = sy-(expendh/2)
    ey = ey+(expendh/2)

    sx = int(max(0, sx))
    sy = int(max(0, sy))
    ex = int(min(w-1, ex))
    ey = int(min(h-1, ey))

    return image[sy:ey, sx:ex, :], sx, sy, ex, ey

def frame_deal(paths, args, lock, counter, total_length):
    # init face detection model
    fd = scrfd.ScrdfFaceDet(0.45,
                            model_path='../utils/face_det_python/models/model.pth',
                            device='cpu' if args.cpu else 'cuda',
                            config='../utils/face_det_python/models/scrfd_500m.py')

    # init facial expression model
    config_file_path = os.path.join(args.mmcls_src_dir, 'configs/fer/mobilenet_v2.py')
    ckpt_path = os.path.join(args.mmcls_src_dir, 'run/fer/mobilenetv2/latest.pth')
    model = init_model(config_file_path, ckpt_path, device='cpu' if args.cpu else 'cuda')
    model.CLASSES = list(dir_label_map)

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
        image_face, isx, isy, iex, iey = get_input_face(image, [sx, sy, ex, ey])

        # inference image with
        result = inference_model(model, image_face, face_rect=None)

        # debug
        print("Predict {}".format(result['pred_label']))
        print(result, model.CLASSES)
        cv2.rectangle(image, (isx, isy), (iex, iey), (255, 0, 0), 10)
        cv2.putText(image, '{}:{:.3f}'.format(dir_label_map[result['pred_label']], result['pred_score']), (sx, sy - 20),
                    0, 2, (0, 0, 255), 2)
        cv2.imshow(dir_label_map[result['pred_label']], image)
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

