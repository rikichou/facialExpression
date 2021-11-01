import os
import glob

# IMG_ROOT_DIR = '/home/ruiming/workspace/pro/facialExpression/data/FacialExpressionOrignialData'
#
# files = glob.glob(os.path.join(IMG_ROOT_DIR, '*/*/*/*.*'))
# angry_files = glob.glob(os.path.join(IMG_ROOT_DIR, '*/*/Angry/*.*'))
# sad_files = glob.glob(os.path.join(IMG_ROOT_DIR, '*/*/Sad/*.*'))
# happy_files = glob.glob(os.path.join(IMG_ROOT_DIR, '*/*/Happy/*.*'))
# neutral_files = glob.glob(os.path.join(IMG_ROOT_DIR, '*/*/Neutral/*.*'))
#
# print("Total images ", len(files))
# print("angry {}, sad {}, happy {}, neutral {}".format(len(angry_files), len(sad_files), len(happy_files), len(neutral_files)))

label_path = '/Users/zhourui/workspace/pro/facialExpression/data/train_mix.txt'

with open(label_path, 'r') as f:
    lines = f.readlines()
    lines = [item.strip() for item in lines]

statistics = {}

cat_sta = {}

for l in lines:
    sample = l.split(';')[0]

    # cat
    cat = sample.split('/')[-2]
    if cat not in cat_sta:
        cat_sta[cat] = 0
    cat_sta[cat] += 1

    # total
    item = os.path.dirname(sample)
    if item not in statistics:
        statistics[item] = 0
    statistics[item] += 1

for k in cat_sta:
    print(k, cat_sta[k])

for k in statistics:
    print(k, statistics[k])