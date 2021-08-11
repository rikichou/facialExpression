import os.path
import xml.dom.minidom

import cv2

image_dir = '/media/ruiming/data/workspace/pro/facialExpression/data/ruiming/images/train/Anger'

def get_xml_info(xmlpath):
    dom = xml.dom.minidom.parse(xmlpath)
    root = dom.documentElement

    node = root.getElementsByTagName('filename')[0]
    filename = node.childNodes[0].data

    node = root.getElementsByTagName('expression')[0]
    expression = node.childNodes[0].data

    node = root.getElementsByTagName('xmin')[0]
    xmin = int(float(node.childNodes[0].data))

    node = root.getElementsByTagName('ymin')[0]
    ymin = int(float(node.childNodes[0].data))

    node = root.getElementsByTagName('xmax')[0]
    xmax = int(float(node.childNodes[0].data))

    node = root.getElementsByTagName('ymax')[0]
    ymax = int(float(node.childNodes[0].data))

    return filename, expression,(xmin,ymin,xmax,ymax)

filename, expression,face_rect = get_xml_info(os.path.join(image_dir, '0ad2dc5cc6f78c7122f6c2f5260338958717275a50520412ebc31dfb.xml'))

img = cv2.imread(os.path.join(image_dir, filename))
sx,sy,ex,ey = face_rect
cv2.rectangle(img, (sx,sy), (ex,ey), (255,0,0), 1)
cv2.imshow(expression, img)
cv2.waitKey(0)
