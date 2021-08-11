import cv2
import pandas as pd
import shutil
import sys,os
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString

imgs_dir = '/media/ruiming/data/workspace/pro/facialExpression/data/ruiming/Manually_Annotated/Manually_Annotated_Images'
labels_dir = '/media/ruiming/data/workspace/pro/facialExpression/data/ruiming/Manually_Annotated_file_lists'

out_dir = '/media/ruiming/data/workspace/pro/facialExpression/data/ruiming/images/val'
if not os.path.exists(out_dir):
       os.makedirs(out_dir)

#label_path = os.path.join(labels_dir, 'training.csv')
label_path = os.path.join(labels_dir, 'validation.csv')
labels = pd.read_csv(label_path)
#labels = labels[:10]

keys = ['subDirectory_filePath', 'face_x', 'face_y', 'face_width',
       'face_height', 'facial_landmarks', 'expression', 'valence', 'arousal']
names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Anger', 'Disgust', 'Contempt', 'None', 'Uncertain', 'Non-face']

print(labels.columns)
print(labels.subDirectory_filePath.value_counts())
print(labels.expression.value_counts())

subDirectory_filePaths = list(labels['subDirectory_filePath'])
face_xs = list(labels['face_x'])
face_ys = list(labels['face_y'])
face_widths = list(labels['face_width'])
face_heights = list(labels['face_height'])
facial_landmarks = list(labels['facial_landmarks'])
expressions = list(labels['expression'])
valences  = list(labels['valence'])
arousals  = list(labels['arousal'])

print(type(subDirectory_filePaths[0]))
print(type(face_xs[0]))
print(type(face_ys[0]))
print(type(face_widths[0]))
print(type(face_heights[0]))
print(type(facial_landmarks[0]))
print(type(expressions[0]))
print(type(valences[0]))
print(type(arousals[0]))

def write_info_to_xml(xmlpath, imgname, shape, face_rect, expression, valence, arousal, folder='train'):
       node_root = Element('annotation')

       node_folder = SubElement(node_root, 'folder')
       node_folder.text = folder

       node_filename = SubElement(node_root, 'filename')
       node_filename.text = imgname

       node_size = SubElement(node_root, 'size')
       node_width = SubElement(node_size, 'width')
       node_width.text = '%s' % shape[0]

       node_height = SubElement(node_size, 'height')
       node_height.text = '%s' % shape[1]

       node_depth = SubElement(node_size, 'depth')
       node_depth.text = '%s' % shape[2]

       # write object info
       sx,sy,ex,ey = face_rect
       node_object = SubElement(node_root, 'object')
       node_name = SubElement(node_object, 'name')
       node_name.text = 'face'
       node_difficult = SubElement(node_object, 'difficult')
       node_difficult.text = '0'
       node_bndbox = SubElement(node_object, 'bndbox')
       node_xmin = SubElement(node_bndbox, 'xmin')
       node_xmin.text = '%s' % sx
       node_ymin = SubElement(node_bndbox, 'ymin')
       node_ymin.text = '%s' % sy
       node_xmax = SubElement(node_bndbox, 'xmax')
       node_xmax.text = '%s' % ex
       node_ymax = SubElement(node_bndbox, 'ymax')
       node_ymax.text = '%s' % ey
       node_expression = SubElement(node_object, 'expression')
       node_expression.text = '%s' % expression
       node_valence = SubElement(node_object, 'valence')
       node_valence.text = '%s' % valence
       node_arousal = SubElement(node_object, 'arousal')
       node_arousal.text = '%s' % arousal

       xml = tostring(node_root, pretty_print=True)
       dom = parseString(xml)

       with open(xmlpath, 'wb') as f:
              f.write(xml)

toatallen = len(subDirectory_filePaths)
not_exist = 0
for idx,p in enumerate(subDirectory_filePaths):
       sx = face_xs[idx]
       sy = face_ys[idx]
       ex = sx+face_widths[idx]-1
       ey = sy+face_heights[idx]-1

       expression = expressions[idx]
       valence = valences[idx]
       arousal = arousals[idx]

       subdir = names[expression]
       out_sub_dir = os.path.join(out_dir, subdir)
       if not os.path.exists(out_sub_dir):
              os.makedirs(out_sub_dir)

       # copy image to new folder
       img_name = os.path.basename(p)
       org_img_path = os.path.join(imgs_dir, p)
       if not os.path.exists(org_img_path):
              print("{} not exist!".format(org_img_path))
              not_exist += 1
              continue
       out_img_path = os.path.join(out_sub_dir, img_name)
       shutil.copy(org_img_path, out_img_path)

       # create xml labels
       image = cv2.imread(org_img_path)
       if image is None:
              print("Failed to read {}!".format(org_img_path))
              continue

       xmlpath = out_img_path.rsplit('.', 1)[0]+'.xml'
       write_info_to_xml(xmlpath, img_name, image.shape, (sx,sy,ex,ey), expression, valence, arousal, folder='val')

       if idx%2000 == 0:
              print("{}/{}".format(idx, toatallen))
print("Deal complete, {} not exist!".format(not_exist))