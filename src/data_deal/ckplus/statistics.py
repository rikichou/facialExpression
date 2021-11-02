import sys
import os
import shutil
import glob


IMG_ROOT_DIR = r'E:\workspace\pro\facialExpression\data\shixisheng_selected\ckplus'
cats = glob.glob(IMG_ROOT_DIR + r'\*\*')

sta = set()
for cat in cats:
    name = os.path.basename(cat)
    sta.add(name)
print(sta)