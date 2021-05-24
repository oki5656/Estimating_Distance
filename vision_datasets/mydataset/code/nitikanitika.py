import cv2
import glob
import os
import re
import numpy as np
from matplotlib import pyplot as plt



########## 設定 ##########
threshold = 2#83は2.5m
nitika_max=1
nitika_normal=255

#original_image_path=os.path.join("..","data_b_kurumadome","depth_g","20*")
#nitiked_image_path=os.path.join("..","data_b_kurumadome","nitika_g")

original_image_path=os.path.join("..","for_ex","output","*.png")
nitiked_image_path=os.path.join("..","for_ex","nitikanitika")
##########################

if not os.path.isdir(nitiked_image_path):
    os.mkdir(nitiked_image_path)

filenames=sorted(glob.glob(original_image_path))
file_num=len(filenames)
print(file_num)
#print("input file path")
for i in range(file_num):
    print("input file path:",filenames[i])
    img = cv2.imread(filenames[i],cv2.IMREAD_GRAYSCALE)

    #print(img.shape)
    #ret, img_thresh = cv2.threshold(img, threshold, nitika_normal, cv2.THRESH_BINARY)
    ret, img_thresh = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)

    #img_thresh=img_thresh+1

    filename_num=re.sub("\\D","",filenames[i])
    filename=filename_num+".png"
    new_name=os.path.join(nitiked_image_path,filename)
    #assert os.path.isdir(nitiked_image_path)
    #print("output file path:",new_name)


    cv2.imwrite(new_name, img_thresh)