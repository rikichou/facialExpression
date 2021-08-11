import sys,os
import pandas as pd
import numpy as np
import shutil

# image info
img_src_dir = '/media/ruiming/data/workspace/pro/facialExpression/data/emotioNet/emotioNet_URLs_Download/downloader_beta1.0/IMAGES/URLsWithEmotionCat_aws_free'
img_out_root_dir = '/media/ruiming/data/workspace/pro/facialExpression/data/emotioNet/emotionet_cat'
if not os.path.exists(img_out_root_dir):
    os.makedirs(img_out_root_dir)

# label info
name_list = np.array(['angrily_disgusted', 'angrily_surprised', 'angry', 'appalled', 'awed', 'disgusted', 'fearful', 'fearfully_angry', 'fearfully_surprised',
             'happily_disgusted', 'happily_surprised', 'happy', 'sad', 'sadly_angry', 'sadly_disgusted', 'surprised'])
file_path = '/home/ruiming/workspace/pro/facialExpression/data/emotioNet/EmotioNet/URLsWithEmotionCat_aws_free.xlsx'

# read and deal label file
df = pd.read_excel(file_path)
data = df.values
for d in data:
    # get all infos
    url = d[0]
    label = d[2:18]

    # get label infos
    labelname = name_list[label == 1]
    if len(labelname) != 1:
        print('{}, multi labels!'.format(labelname))
    labelname = labelname[0]
    img_out_dir = os.path.join(img_out_root_dir, labelname)
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)

    # get src image info
    imgname = os.path.basename(url)
    img_src_path = os.path.join(img_src_dir, imgname)
    img_dst_path = os.path.join(img_out_dir, imgname)

    # copy image
    if not os.path.exists(img_src_path):
        print("Warning ! {}".format(img_src_path))
        continue
    shutil.copy(img_src_path, img_dst_path)

