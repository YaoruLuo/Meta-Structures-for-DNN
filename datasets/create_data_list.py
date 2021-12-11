# -*- coding:utf-8 -*-

import os
import sys
import glob

currentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(currentdir)

image_dir = ".."

mask_dir = ".."
txt_dir = currentdir + "/datasets/txt/er/train.txt"


dir_list = os.listdir(image_dir)

img_list = glob.glob(os.path.join(image_dir, "*.tif"))

is_test = False

for image_path in img_list:
    image_name = os.path.split(image_path)[1].split('.')[0]
    print(image_name)
    mask_path = os.path.join(mask_dir, image_name + '.tif')
    if is_test == False:
        with open(txt_dir, "a+") as fid:
            fid.write(image_path + " " + mask_path + "\n")
    else:
        with open(txt_dir, "a+") as fid:
            fid.write(image_path + "\n")

