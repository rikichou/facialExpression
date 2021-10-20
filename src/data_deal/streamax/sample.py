import os
import shutil
import glob


IMG_ROOT_DIR = r'/home/ruiming/workspace/pro/facialExpression/data/select/finish2/streamax_0825/train/happy'
OUT_IMG_ROOT_DIR = r'/home/ruiming/workspace/pro/facialExpression/data/select/streamax'
if not os.path.exists(OUT_IMG_ROOT_DIR):
    os.makedirs(OUT_IMG_ROOT_DIR)

imgs = glob.glob(IMG_ROOT_DIR+'/*.jpg')

statistics = {}
for img in imgs:
    name = os.path.basename(img)
    v = name[:6]

    if v not in statistics:
        statistics[v] = []

    statistics[v].append(img)

items = statistics.items()
soted = sorted(items, key=lambda x:len(x[1]), reverse=True)

print("Total {} images, {} videos, max {}, min {}".format(len(imgs), len(statistics), len(statistics[soted[0][0]]), len(statistics[soted[-1][0]])))
print(soted[0][0], statistics[soted[0][0]])