import os.path
import sys
import shutil
import glob

src_img_dir = '/home/ruiming/workspace/pro/facialExpression/data/JAFFE/jaffeim.ages/jaffe'
dst_img_root_dir = '/home/ruiming/workspace/pro/facialExpression/data/JAFFE/cat'
if not os.path.exists(dst_img_root_dir):
    os.makedirs(dst_img_root_dir)

labelmap = {'SA':'Sad', 'SU':'Surprise', 'HA':'Happy', 'AN':'Angry', 'DI':'Disgusted', 'NE':'Neutral', 'FE':'Fearful'}

category_statistics = {}
imgsfile = glob.glob(src_img_dir+'/*.tiff')
for imgfile in imgsfile:
    imgname = os.path.basename(imgfile)
    label = imgname.split('.')[1][:2]
    # statistics
    catname = labelmap[label]
    if catname not in category_statistics:
        category_statistics[catname] = 0
    category_statistics[catname]  += 1

    # copy image
    dst_img_dir = os.path.join(dst_img_root_dir, catname)
    if not os.path.exists(dst_img_dir):
        os.makedirs(dst_img_dir)
    dst_img_path = os.path.join(dst_img_dir, imgname)
    shutil.copy(imgfile, dst_img_path)

print(category_statistics)


