import cv2
import glob
import os
import re
import numpy as np
from matplotlib import pyplot as plt



########## 設定 ##########
path_w=os.path.join("..","val.txt")
src_image_path=os.path.join("..","for_ex","rgb_resized_png","*.png")#ディレクトリ名には数字を入れない
##########################


filenames=sorted(glob.glob(src_image_path))
file_num=len(filenames)
print(file_num)

with open(path_w,mode="w") as f:

    for i in range(file_num):
        #img = cv2.imread(filenames[i],0)
        #print(filenames[i])       
        filename_num=re.sub("\\D","",filenames[i])
        rgb_filename=filename_num+".png"
        mask_filename=filename_num+".png"
        writing_line=rgb_filename+","+mask_filename+"\n"
        #new_name=os.path.join(nitiked_image_path,filename)
        print(writing_line,end="")

        f.write(writing_line)
    
