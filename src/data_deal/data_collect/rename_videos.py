import os
import glob
import time

VIDEO_DIR = r'G:\pro\facialexpression\data\youtube\video'
video_paths = glob.glob(os.path.join(VIDEO_DIR, '*.mp4'))

count = 1
for vpath in video_paths:
    new_video_name = 'download_{:07d}.mp4'.format(count)
    new_video_name_path = os.path.join(os.path.dirname(vpath), new_video_name)
    os.rename(vpath, new_video_name_path)
    count += 1