import cv2
import glob
import os
import re
import numpy as np
from matplotlib import pyplot as plt



################################ 設定 ################################
src_path=os.path.join("..","for_ex","*.JPG")
output_path=os.path.join("..","for_ex","rgb_resized_png")#ディレクトリ名には数字を入れない
resize_rario=0.5
######################################################################
if not os.path.exists(output_path):
    os.mkdir(output_path)

filenames=sorted(glob.glob(src_path))
file_num=len(filenames)
print(file_num)


for i in range(file_num):
    img = cv2.imread(filenames[i],1)
    print(filenames[i])     
    height = img.shape[0]
    width = img.shape[1]
    img2 = cv2.resize(img , (int(width*resize_rario), int(height*resize_rario)))
    filename_num=re.sub("\\D","",filenames[i])
    new_filename=filename_num+".png"
    new_path=os.path.join(output_path,new_filename)
    cv2.imwrite(new_path, img2)