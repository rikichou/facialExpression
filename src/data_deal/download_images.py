import sys
import os.path
import urllib
from urllib import request
import glob

if __name__== '__main__':

    if len(sys.argv) != 2:
        print("Invalid command parameters! ./download_images.py SRC_URL_DIR")

    #src_urls_dir = '/media/ruiming/data/workspace/pro/facialExpression/data/emotioNet/EmotioNet/emotioNet_challenge_files_server_challenge_1.2_aws'
    #src_urls_dir = '/media/ruiming/data/workspace/pro/facialExpression/data/emotioNet/emotioNet_URLs_Download/downloader_beta1.0'
    src_urls_dir = sys.argv[1]
    out_images_base_dir = '/media/ruiming/data/workspace/pro/facialExpression/data/emotioNet/EmotioNet/images'
    txt_files = glob.glob(os.path.join(src_urls_dir, '*.txt'))

    files_num = len(txt_files)
    for fidx,file_path in enumerate(txt_files):
        file_name = os.path.basename(file_path)
        prefix,ext = os.path.splitext(file_name)
        # create out dir
        out_images_dir = os.path.join(out_images_base_dir, prefix)
        if not os.path.exists(out_images_dir):
            os.makedirs(out_images_dir)
        # read urls file and download
        with open(file_path, 'r') as fp:
            for idx,l in enumerate(fp):
                l = l.strip().split()[0]
                out_image_path = os.path.join(out_images_dir, os.path.basename(l))
                urllib.request.urlretrieve(l, filename=out_image_path)
                if idx%1 == 0:
                    print("{}/{} -- {}".format(fidx, files_num, idx))
