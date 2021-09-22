import os
from tqdm import tqdm
from multiprocessing import Pool
from pytube import YouTube
import argparse

num_worker = 10

def download(vid):
    # check if video is exist
    out_video_path = os.path.join(out_video_dir, vid + '.mp4')
    if os.path.exists(out_video_path):
        return

    # download vide
    base_url = 'https://www.youtube.com/watch?v='
    video_url = base_url + vid
    yt = YouTube(video_url)

    try:
        handle = yt.streams.get_by_resolution('720p')
        if not handle:
            handle = yt.streams.get_highest_resolution()
        # size check
        video_size = handle.filesize/(1024*1024)
        if video_size<=400:
            handle.download(output_path=out_video_dir, filename=vid + '.mp4')
        else:
            print("video is too large: {} MB, skip".format(video_size))
    except:
        print("Some error in ", vid)

def parse_args():
    parser = argparse.ArgumentParser(description='download videos')
    parser.add_argument(
        'vids_file_path', type=str, help='file include vids')
    parser.add_argument(
        'out_video_dir', type=str, help='out name')

    args = parser.parse_args()
    return args

def main():
    # get all vid list
    with open(vids_file_path, 'r') as fp:
        vids = [item.strip() for item in fp.readlines()]

    with Pool(num_worker) as pool:
        r = list(tqdm(pool.imap(
            download,vids)))

if __name__ == '__main__':
    args = parse_args()
    vids_file_path = args.vids_file_path
    out_video_dir = args.out_video_dir
    if not os.path.exists(out_video_dir):
        os.makedirs(out_video_dir)

    main()