import sys,os
import shutil
import cv2
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString

label_path = '/media/ruiming/data/workspace/pro/facialExpression/data/ExpW/data/label/label.lst'
src_img_dir = '/media/ruiming/data/workspace/pro/facialExpression/data/ExpW/data/image/origin'
dst_img_root_dir = '/media/ruiming/data/workspace/pro/facialExpression/data/ExpW/data/image/cat'
if not os.path.exists(dst_img_root_dir):
    os.makedirs(dst_img_root_dir)

# label map
label_map = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'sad', 5:'surprise', 6:'neutral'}

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
with open(label_path, 'r') as fp:
    for line in fp:
        infos = line.strip().split()

        # filter low qua face
        face_prob = float(infos[6])
        if face_prob < 30:
            continue

        # statistic
        cat_name = label_map[int(infos[7])]
        if cat_name not in label_statistics:
            label_statistics[cat_name] = 0
        label_statistics[cat_name] += 1

        # check if image name exist
        imgname = infos[0]
        if imgname not in label_dict:
            label_dict[imgname] = {}
        else:
            # statistics
            if cat_name not in label_dup_statistics:
                label_dup_statistics[cat_name] = 0
            label_dup_statistics[cat_name] += 1
        # check if cat exist
        if cat_name not in label_dict[imgname]:
            label_dict[imgname][cat_name] = []
        # save info
        face_box_top = int(infos[2])
        face_box_left = int(infos[3])
        face_box_right = int(infos[4])
        face_box_bottom = int(infos[5])
        label_dict[imgname][cat_name].append([face_box_left, face_box_top, face_box_right, face_box_bottom])

print(label_statistics)
print(label_dup_statistics)

total = len(label_dict)
for idx,imgname in enumerate(label_dict):
    cat_infos = label_dict[imgname]

    # image info
    src_img_path = os.path.join(src_img_dir, imgname)
    image = cv2.imread(src_img_path)
    if image is None:
        continue

    for cat_name in cat_infos:
        # get dst path from cat_name
        dst_img_dir = os.path.join(dst_img_root_dir, cat_name)
        if not os.path.exists(dst_img_dir):
            os.makedirs(dst_img_dir)
        dst_img_path = os.path.join(dst_img_dir, imgname)
        # copy image
        shutil.copy(src_img_path, dst_img_path)

        # get out_xml_path from imgname
        img_prefix,_ = os.path.splitext(imgname)
        xml_name = img_prefix+'.xml'
        out_xml_path = os.path.join(dst_img_dir, xml_name)
        infos = cat_infos[cat_name]
        # save xml
        write_info_to_xml(out_xml_path, imgname, image.shape, infos, cat_name, folder='whatever')

    if idx%1000 == 0:
        print("{}/{}".format(idx,total))



