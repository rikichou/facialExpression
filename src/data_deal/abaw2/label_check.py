import os
import sys
import glob

ANNS_ROOT_DIR = r'G:\pro\facialexpression\data\abaw2\annotations\EXPR_Set'
anns = glob.glob(ANNS_ROOT_DIR+r'\*\*.txt')

def get_mult_anns(anns):
    mutl_anns = []
    for ann in anns:
        if 'left' in ann:
            key_word = '_left'
        elif 'right' in ann:
            key_word = '_right'
        else:
            continue
        mutl_anns.append(ann.replace(key_word, ''))
        mutl_anns.append(ann)
    return mutl_anns


to_del = get_mult_anns(anns)

for item in to_del:
    if os.path.exists(item):
        print('del ', item)
        os.remove(item)