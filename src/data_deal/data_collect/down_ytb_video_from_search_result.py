import requests
from pytube import YouTube
from tqdm import tqdm

YouTube('https://youtu.be/2lAe1cqCOXo').streams.first().download()

out_vid_file_path = './20210916_1936.txt'
out_video_dir = r'G:\pro\facialexpression\data\youtube\video'
max_num = 50

next_page = 'CN4CEAA'

# search keyword
keywords = '电影发怒生气片段'
api_key = 'AIzaSyC7ihXkK4J-k3-j2gn4o_h-BFj2vlbyz0A'
api_uri = 'https://www.googleapis.com/youtube/v3/'

if next_page:
    url = f'{api_uri}search?key={api_key}&maxResults={max_num}&part=id&type=video&q={keywords}&pageToken={next_page}'
else:
    url = f'{api_uri}search?key={api_key}&maxResults={max_num}&part=id&type=video&q={keywords}'

r2 = requests.get(url)
result = r2.json()

base_url = 'https://www.youtube.com/watch?v='
while True:
    if 'nextPageToken' in result and result['nextPageToken']:
        next_page = result['nextPageToken']
    else:
        next_page = None

    for item in tqdm(result['items']):
        vid = item['id']['videoId']
        video_url = base_url+vid
        print(video_url)
        yt = YouTube(video_url)

        handle = yt.streams.get_by_resolution('720p')
        if handle:
            handle.download(output_path=out_video_dir)
        else:
            yt.streams.get_highest_resolution().download(output_path=out_video_dir)

    if next_page == None:
        break
    else:
        url = f'{api_uri}search?key={api_key}&maxResults={max_num}&part=id&type=video&q={keywords}&pageToken={next_page}'

        r2 = requests.get(url)
        result = r2.json()