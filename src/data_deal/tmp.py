import sys
print(sys.path)

from utils import common


image_path = '/train_tmp/fer/select/finish2/ExpW/train/angry_light/mad_old_79.jpg'

common.show_facerect_with_txtfile(image_path, save_path='/home/ruiming/workspace/tmp/show1.jpg')