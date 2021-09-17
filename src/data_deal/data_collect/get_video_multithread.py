import os
from tqdm import tqdm
from multiprocessing import Pool
from pytube import YouTube

num_worker = 10

vids_file_path = r'G:\pro\facialexpression\data\youtube\20210916_1936_dianyingfanushengqipianduan.txt'
out_video_dir = r'G:\pro\facialexpression\data\youtube\video_multi'
if not os.path.exists(out_video_dir):
    os.makedirs(out_video_dir)

def download(vid):
    # check if video is exist
    out_video_path = os.path.join(out_video_dir, vid + '.mp4')
    if os.path.exists(out_video_path):
        return

    # download vide
    base_url = 'https://www.youtube.com/watch?v='
    video_url = base_url + vid
    yt = YouTube(video_url)
    handle = yt.streams.get_by_resolution('720p')
    if handle:
        handle.download(output_path=out_video_dir, filename=vid + '.mp4')
    else:
        yt.streams.get_highest_resolution().download(output_path=out_video_dir, filename=vid + '.mp4')

def main():
    # get all vid list
    with open(vids_file_path, 'r') as fp:
        vids = [item.strip() for item in fp.readlines()]

    with Pool(num_worker) as pool:
        r = list(tqdm(pool.imap(
            download,vids)))

if __name__ == '__main__':
    main()