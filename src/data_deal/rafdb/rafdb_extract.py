import sys,os
import shutil
import cv2
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString

label_expression_path = '/home/ruiming/workspace/pro/facialExpression/data/RAFDB/basic/EmoLabel/list_patition_label.txt'
label_bbox_dir = '/home/ruiming/workspace/pro/facialExpression/data/RAFDB/basic/Annotation/boundingbox'
src_img_dir = '/home/ruiming/workspace/pro/facialExpression/data/RAFDB/basic/Image/original'
dst_img_root_dir = '/home/ruiming/workspace/pro/facialExpression/data/RAFDB/cat'
if not os.path.exists(dst_img_root_dir):
    os.makedirs(dst_img_root_dir)

# label map
label_map = {6:'angry', 3:'disgust', 2:'fear', 4:'happy', 5:'sad', 1:'surprise', 7:'neutral'}

def write_info_to_xml(xmlpath, imgname, shape, face_rects, expression, folder='whatever'):
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
    for face_rect in face_rects:
        sx, sy, ex, ey = face_rect
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

    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)

    with open(xmlpath, 'wb') as f:
        f.write(xml)

label_dict = {}
label_statistics = {}
label_dup_statistics = {}
with open(label_expression_path, 'r') as fp:
    for idx,line in enumerate(fp):
        infos = line.strip().split()

        # get expression label
        label = int(infos[1])
        cat_name = label_map[label]
        out_img_dir = os.path.join(dst_img_root_dir, cat_name)
        if not os.path.exists(out_img_dir):
            os.makedirs(out_img_dir)

        # get imagename and copy image
        imgname = infos[0]
        src_img_path = os.path.join(src_img_dir, imgname)
        if not os.path.exists(src_img_path):
            print("{} not exist!".format(src_img_path))
            continue
        image = cv2.imread(src_img_path)
        if image is None:
            print('Failed to read image {}'.format(src_img_path))
            continue
        dst_img_path = os.path.join(out_img_dir, imgname)
        shutil.copy(src_img_path, dst_img_path)

        # get bbox info
        imgprefix,_ = os.path.splitext(imgname)
        label_bbox_path = os.path.join(label_bbox_dir, imgprefix+'_boundingbox.txt')
        if not os.path.exists(label_bbox_path):
            print("!label {} not exist!".format(label_bbox_path))
            continue
        with open(label_bbox_path, 'r') as bfp:
            bbox = bfp.readlines()[0].strip().split()
            sx = int(float(bbox[0]))
            sy = int(float(bbox[1]))
            ex = int(float(bbox[2]))
            ey = int(float(bbox[3]))

        # statistic
        if cat_name not in label_statistics:
            label_statistics[cat_name] = 0
        label_statistics[cat_name] += 1

        # get out_xml_path from imgname
        xml_name = imgprefix+'.xml'
        out_xml_path = os.path.join(out_img_dir, xml_name)
        # save xml
        write_info_to_xml(out_xml_path, imgname, image.shape, [[sx,sy,ex,ey]], cat_name, folder='whatever')

        if idx%1000 == 0:
            print("{}".format(idx))

print(label_statistics)
