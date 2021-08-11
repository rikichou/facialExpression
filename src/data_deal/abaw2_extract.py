import os
import cv2
import glob
import imageio

videos_root_dir = '/home/ruiming/workspace/pro/facialExpression/data/ABAW2/videos/batch1_batch2'
labels_root_dir = '/home/ruiming/workspace/pro/facialExpression/data/ABAW2/annotation/annotations/annotations/EXPR_Set/train_and_validation'

# get all labels which has only one subject int the vedio
labels_name = os.listdir(labels_root_dir)
valid_labels_name = []
invalid_labels_name = []
for lname in labels_name:
    if 'left' in lname or 'right' in lname:
        orglname = lname.rsplit('_', maxsplit=1)[0]+'.txt'
        invalid_labels_name.append(orglname)
    else:
        valid_labels_name.append(lname)
print(invalid_labels_name)
print("==============================")

videos_name = os.listdir(videos_root_dir)
for vname in videos_name:
    # check if invalid
    if '.mp4' not in vname and '.avi' not in vname:
        print("Invalid video file {}".format(vname))
        continue
    lname = os.path.splitext(vname)[0]+'.txt'
    if lname in invalid_labels_name:
        print("Skip invalid video {}".format(vname))
        continue
    if lname in valid_labels_name:
        # get video infos
        vpath = os.path.join(videos_root_dir, vname)
        cap = cv2.VideoCapture(vpath)
        total_frames = int(cap.get(7))
        fps = int(cap.get(5))
        count = 0
        while True:
            ret,frame = cap.read()
            if ret is False:
                break
            count += 1
        # vid = imageio.get_reader(vpath, 'ffmpeg')
        # total_frames = len(vid)

        # get label infos
        lpath = os.path.join(labels_root_dir, lname)
        with open(lpath, 'r') as fp:
            lines = fp.readlines()
            print(len(lines)-1, total_frames, count, vname)



    # baselname = os.path.basename(lpath)
    # if baselname in invalid_labels_name:
    #     continue
    # # get video path
    # basevname = baselname[]
    # video_path = '/home/ruiming/workspace/pro/facialExpression/data/ABAW2/videos/batch1/23-24-1920x1080.mp4'
    #
    # cap = cv2.VideoCapture(video_path)
    #
    # if not os.path.exists(video_path):
    #     print("not exist {}".format(video_path))
    #
    # print(cap.get(7), cap.get(5))