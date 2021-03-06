import sys
sys.path.append('..')
import os
import numpy as np
import argparse
import glob
import time
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
import xml.dom.minidom
import cv2
from utils import common


def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument(
        'src_dir', type=str, help='root directory for the frames')
    parser.add_argument(
        '--level', type=int, default=4, help='dir level')
    parser.add_argument(
        '--ext', type=str, default='jpg', help='dir level')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=16,
        help='number of workers to build rawframes')
    args = parser.parse_args()

    return args

def get_xml_info(xmlpath):
    dom = xml.dom.minidom.parse(xmlpath)
    root = dom.documentElement

    # get h and w
    size_node = root.getElementsByTagName('size')[0]
    width_node = size_node.getElementsByTagName('width')[0]
    width = int(float(width_node.childNodes[0].data))
    height_node = size_node.getElementsByTagName('height')[0]
    height = int(float(height_node.childNodes[0].data))

    nodes = root.getElementsByTagName('object')
    ret = []
    for object_node in nodes:
        name_node = object_node.getElementsByTagName('name')[0]
        name = name_node.childNodes[0].data
        if name != 'face':
            print("name is {} !".format(name))
            continue

        bndbox_node = object_node.getElementsByTagName('bndbox')[0]
        xmin_node = bndbox_node.getElementsByTagName('xmin')[0]
        xmin = int(float(xmin_node.childNodes[0].data))

        ymin_node = bndbox_node.getElementsByTagName('ymin')[0]
        ymin = int(float(ymin_node.childNodes[0].data))

        xmax_node = bndbox_node.getElementsByTagName('xmax')[0]
        xmax = int(float(xmax_node.childNodes[0].data))

        ymax_node = bndbox_node.getElementsByTagName('ymax')[0]
        ymax = int(float(ymax_node.childNodes[0].data))

        ret.append([xmin,ymin,xmax,ymax])

    return ret,height,width

def get_new_rect(rects, h, w):
    ret = []
    for rect in rects:
        xmin,ymin,xmax,ymax = rect
        face_h = ymax - ymin
        face_w = xmax - xmin

        exp_h = int(face_h*0.2)
        ymin = max(0, ymin-exp_h)
        ret.append([xmin,ymin,xmax,ymax])
    return ret


def deal(f):
    # get xml file path
    img_dir = os.path.basename(os.path.dirname(f))
    xml_dir = 'xml'+img_dir[3:]
    xml_name = os.path.basename(f)[:-3]+'xml'
    xml_path = os.path.join(os.path.dirname(os.path.dirname(f)), xml_dir, xml_name)
    if not os.path.exists(xml_path):
        print('Can not find xml : ', xml_path)
        return True

    # get xml info
    face_rects,height,width = get_xml_info(xml_path)
    if len(face_rects) != 1:
        # check if have no face, or multi face
        return True
    new_face_rects = get_new_rect(face_rects, height, width)

    # save info
    txt_save_path = os.path.join(os.path.dirname(f), os.path.basename(f)[:-3]+'facerect')
    common.save_lists_to_txtfile(new_face_rects, txt_save_path)
    # debug
    # for rect in face_rects:
    #
    #     image = cv2.imread(f)
    #     image = cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,0,255), 2)
    #     cv2.imshow('1', image)
    #     key = cv2.waitKey(0)
    #     if key==ord('q'):
    #         return True
    return True

def main():
    args = parse_args()

    # read all npy
    st = time.time()
    print("Start to get all {} files".format(args.ext))
    files = glob.glob(args.src_dir + str(Path('/*' * args.level)) + '.' +
              args.ext)
    print("Glob get {} files, Time cost {} seconds".format(len(files), time.time()-st))

    with Pool(args.num_worker) as pool:
        r = list(tqdm(pool.imap(
            deal,files)))

if __name__ == '__main__':
    main()






