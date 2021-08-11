import sys,os,shutil
import numpy as np

# src image dir and dst image dir
img_root_dir = '/media/ruiming/data/workspace/pro/facialExpression/data/emotioNet/emotioNet_URLs_Download/downloader_beta1.0/IMAGES'
label_dir = '/media/ruiming/data/workspace/pro/facialExpression/data/emotioNet/EmotioNet/emotioNet_challenge_files_server_challenge_1.2_aws'
dst_root_dir = '/media/ruiming/data/workspace/pro/facialExpression/data/emotioNet/EmotioNet/images'

labels_name = os.listdir(label_dir)
#valid_ann_idx = np.array([1, 2, 4, 5,6, 9, 12, 17, 20, 25, 26, 43]) - 1
valid_ann_idx = np.array([1, 2, 4, 5, 6, 9, 10, 12, 15, 17, 18, 20, 24, 25, 26, 28, 43, 51, 52, 53, 54, 55, 56]) - 1

# map between expression category and AUs
expression_aus_map_dict = {'Happy':[12, 25], 'Sad':[4, 15], 'Fearful':[1, 4, 20, 25],
                           'Angry':[4, 7, 24], 'Surprised':[1, 2, 25, 26], 'Disgusted':[9, 10, 17],
                           'Happily_sad':[4, 6, 12, 25], 'Happily_surpd':[1, 2, 12, 25], 'Happily_disgd':[10, 12, 25],
                           'Sadly_fearful':[1, 4, 15, 25], 'Sadly_angry':[4, 7, 15], 'Sadly_surprised':[1, 4, 25, 26],
                           'Sadly_disgusted':[4, 10], 'Fearfully_angry':[4, 20, 25], 'Fearfully_surpd':[1, 2, 5, 20, 25],
                           'Fearfully_disgd':[1, 4, 10, 20, 25], 'Angrily_surprised':[4, 25, 26], 'Disgd_surprised':[1, 2, 5, 10],
                           'Happily_fearful':[1, 2, 12, 25, 26], 'Angrily_disgusted':[4, 10, 17], 'Awed':[1, 2, 5, 25],
                           'Appalled':[4, 9, 10], 'Hatred':[4, 7, 10]}

def aus_to_str_dict(map_dict):
    ret_dict = {}
    for k in map_dict:
        idx_list = np.array(map_dict[k])-1
        hash_list = np.zeros(60)
        hash_list[idx_list] = 1
        hash_list = hash_list[valid_ann_idx]
        hash_list = [str(int(x)) for x in hash_list]
        aus_str = ''.join(hash_list)
        ret_dict[aus_str] = k
    return ret_dict


ausstr_expression_map_dict = aus_to_str_dict(expression_aus_map_dict)
print(ausstr_expression_map_dict)

def get_hash(aus):
    lable_list = np.array(aus)
    hash = lable_list[valid_ann_idx]
    hashstr = ''.join(hash)

    return hashstr

def cat_images(l_path, src_img_dir):
    with open(l_path, 'r') as fp:
        for line in fp:
            # label info split
            line = line.strip()
            infos = line.split()

            # get image information from url
            img_url = infos[0]
            src_img_name = os.path.basename(img_url)
            src_img_path = os.path.join(src_img_dir, src_img_name)

            # get label information from annotation infos
            aus = infos[2:]

            # get hash str
            hash_str = get_hash(aus)


for idx,l in enumerate(labels_name):
    if '.txt' not in l:
        continue
    # open and deal label file
    l_path = os.path.join(label_dir, l)
    name,ext = os.path.splitext(l)
    img_src_dir = os.path.join(img_root_dir, name)
    cat_images(l_path, img_src_dir)

    break