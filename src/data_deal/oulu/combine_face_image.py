import os
import numpy as np
import glob
import cv2
import argparse
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm

def image_write(path_A, path_B, path_AB, img_size):
    im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_A = cv2.resize(im_A, (img_size, img_size))
    im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_B = cv2.resize(im_B, (img_size, img_size))
    im_AB = np.concatenate([im_A, im_B], 1)
    cv2.imwrite(path_AB, im_AB)

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='./NI/Strong')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='./VL/Strong')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='./NI_VL')
parser.add_argument('--level', dest='level', help='directory level', type=int, default=3)
parser.add_argument('--img_size', dest='img_size', help='image size', type=int, default=224)
parser.add_argument('--num_worker', dest='num_worker', help='worker num', type=int, default=8)
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))

if not os.path.exists(args.fold_AB):
    os.makedirs(args.fold_AB)

fullpath_list = glob.glob(args.fold_A + str(Path('/*' * args.level)) + '.jpeg')
pool=Pool()

def deal_data(info):
    global args
    path_A,vid = info
    name_A = '/'.join(path_A.split('/')[-3:])
    name_AB = '_'.join(path_A.split('/')[-3:])
    path_B = os.path.join(args.fold_B, name_A)
    if os.path.isfile(path_A) and os.path.isfile(path_B):
        path_AB = os.path.join(args.fold_AB, name_AB)
        image_write(path_A, path_B, path_AB, args.img_size)

with Pool(args.num_worker) as pool:
    r = list(
        tqdm.tqdm(
            pool.imap(deal_data,
                      zip(fullpath_list, range(len(fullpath_list))))))