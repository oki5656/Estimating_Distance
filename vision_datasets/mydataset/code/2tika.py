import cv2
import glob
import os
import re
import numpy as np
from matplotlib import pyplot as plt



########## 設定 ##########
threshold = 15#83は2.5m
nitika_max=1
nitika_normal=255

original_image_path=os.path.join("..","annotations","*.png")
nitiked_image_path=os.path.join("..","nitiked")
if not os.path.exists(nitiked_image_path):
    os.mkdir(nitiked_image_path)

#original_image_path=os.path.join("..","data_e_pura-bar","depth_j","21*")
#nitiked_image_path=os.path.join("..","data_e_pura-bar","nitika_j")
##########################

os.path.isdir("original_image_path")
os.path.isdir("nitiked_image_path")

filenames=sorted(glob.glob(original_image_path))
file_num=len(filenames)
print(file_num)
#print("input file path")
for i in range(file_num):
    print("input file path:",filenames[i])
    img = cv2.imread(filenames[i],cv2.IMREAD_GRAYSCALE)

    #print(img.shape)
    #ret, img_thresh = cv2.threshold(img, threshold, nitika_normal, cv2.THRESH_BINARY)
    ret, img_thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    #img_thresh=img_thresh+1

    filename_num=re.sub("\\D","",filenames[i])
    filename=filename_num+".png"
    new_name=os.path.join(nitiked_image_path,filename)
    #assert os.path.isdir(nitiked_image_path)
    #print("output file path:",new_name)


    cv2.imwrite(new_name, img_thresh)#,img_thresh