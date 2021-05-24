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

original_image_path=os.path.join("..","for_ex","calc_50cm","*.png")
#nitiked_image_path=os.path.join("..","for_ex","nitikanitika")
##########################

filenames=sorted(glob.glob(original_image_path))
file_num=len(filenames)
print(file_num)
#print("input file path")
c=0
for i in range(file_num):
    print("input file path:",filenames[i])
    img = cv2.imread(filenames[i],cv2.IMREAD_GRAYSCALE)
    print(np.unique(img))
    a=np.sum(img==0)
    b=np.sum(img==38)
    print(b/(a+b))
    c+=b/(a+b)
    print()
print(c/file_num)#0.00454
