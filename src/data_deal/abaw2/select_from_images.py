import sys
import os
import glob
import shutil
import tqdm
from multiprocessing import Pool

# out info
OUT_ROOT_DIR = r'G:\pro\facialexpression\data\abaw2\expression'
#label_map =  {0:'Neutral',1:'Angry',2:'Disgust',3:'Fear',4:'Happy',5:'Sad',6:'Surprise'}
label_map =  {1:'Angry',5:'Sad'}

# image info
IMAGES_ROOT_DIR = r'G:\pro\facialexpression\data\abaw2\images'

# anns info
ANN_ROOT_DIR = r'G:\pro\facialexpression\data\abaw2\annotations\EXPR_Set\all'
anns = glob.glob(ANN_ROOT_DIR+'\*.txt')

# worker
num_worker = 10

def deal(params):
    ann_file_path,id = params
    with open(ann_file_path) as fp:
        lines = [line.strip() for line in fp.readlines()]
        # remove first line
        lines = lines[1:]

    # find image dir
    dname = os.path.basename(ann_file_path).replace('.txt', '')
    src_img_dir = os.path.join(IMAGES_ROOT_DIR, dname)
    if not os.path.exists(src_img_dir):
        print("Can not fount {} for {}".format(src_img_dir, ann_file_path))
        return

    # copy image to dst dir
    labels = [int(item) for item in lines]
    for idx,label in enumerate(labels):
        if label not in label_map:
            continue
        # create out dir
        img_name = 'img_{:05d}.jpg'.format(idx + 1)
        out_img_dir = os.path.join(OUT_ROOT_DIR, label_map[label])
        if not os.path.exists(out_img_dir):
            os.makedirs(out_img_dir)
        dst_img_name = dname + '_' + img_name
        dst_img_path = os.path.join(out_img_dir, dst_img_name)

        # get src image path
        src_img_path = os.path.join(src_img_dir, img_name)
        if not os.path.exists(src_img_path):
            continue
        shutil.copy(src_img_path, dst_img_path)

if __name__ == '__main__':
    with Pool(num_worker) as pool:
        r = list(tqdm.tqdm(pool.imap(
            deal,
            zip(anns, range(len(anns))))))