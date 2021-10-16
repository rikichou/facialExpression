import sys
import os
import glob

txt_list = ['20211003_2027_dafaleiting.txt','20211003_2027_fennuxinteng.txt','20211003_2027_meitoujinsuo.txt',
            '20211003_2027_naoxiuchengnu.txt']
txt_list = glob.glob('./20211015*')
print(txt_list)

out_list = '20211015.txt'

all_lines = []
for t in txt_list:
    with open(t, 'r') as fp:
        all_lines.extend([item.strip() for item in fp.readlines()])
all_sets = set(all_lines)

print(len(all_lines), len(all_sets))

all_out = [item+'\n' for item in all_sets]

with open(out_list, 'w') as fp:
    fp.writelines(all_out)
