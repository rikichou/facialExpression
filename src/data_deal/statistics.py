import os
import glob

IMG_ROOT_DIR = '/home/ruiming/workspace/pro/facialExpression/data/FacialExpressionOrignialData'

files = glob.glob(os.path.join(IMG_ROOT_DIR, '*/*/*/*.*'))
angry_files = glob.glob(os.path.join(IMG_ROOT_DIR, '*/*/Angry/*.*'))
sad_files = glob.glob(os.path.join(IMG_ROOT_DIR, '*/*/Sad/*.*'))
happy_files = glob.glob(os.path.join(IMG_ROOT_DIR, '*/*/Happy/*.*'))
neutral_files = glob.glob(os.path.join(IMG_ROOT_DIR, '*/*/Neutral/*.*'))

print("Total images ", len(files))
print("angry {}, sad {}, happy {}, neutral {}".format(len(angry_files), len(sad_files), len(happy_files), len(neutral_files)))