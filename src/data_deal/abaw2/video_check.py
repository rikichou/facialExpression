import sys
import os
import glob
import cv2

# video info
VIDEO_ROOT_DIR = r'G:\pro\facialexpression\data\abaw2\videos'

# anns info
ANN_ROOT_DIR = r'G:\pro\facialexpression\data\abaw2\annotations\EXPR_Set\all'
anns = glob.glob(ANN_ROOT_DIR+'\*.txt')

for ann_file_path in anns:
    with open(ann_file_path) as fp:
        lines = [line.strip() for line in fp.readlines()]
        line_len = len(lines) - 1
    # find video
    video_name = os.path.basename(ann_file_path).replace('txt', 'mp4')
    video_path = os.path.join(VIDEO_ROOT_DIR, video_name)
    if not os.path.exists(video_path):
        video_name = os.path.basename(ann_file_path).replace('txt', 'avi')
        video_path = os.path.join(VIDEO_ROOT_DIR, video_name)
    # openvideo and get frame info
    #vr = mmcv.VideoReader(ann_file_path)
    vcap = cv2.VideoCapture(video_path)
    frame_cnt = vcap.get(cv2.CAP_PROP_FRAME_COUNT)

    print("Line:{} FrameCount:{}, last {}, line0 {}".format(line_len, frame_cnt, lines[-1], lines[0]))


