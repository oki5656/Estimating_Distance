"""App"""
import os
import glob
import cv2
import streamlit as st#pip install streamlit==0.48.1
from PIL import Image
import test_semseg
import numpy as np

st.title("Measuring Distance App")
st.write('\n\n')
st.write("How to use the app?")
st.write("1:upload image from left bar\n")
st.write("2:if camera range is not vertical=50,horizontal=63, change it from left bar\n")
st.write("3:automaticaly output result(distance)")
st.write("")


################## サイドバー ##################
st.sidebar.title('Upload an image here')
v,h=0,0
st.sidebar.title('Change camera range from here')
if st.sidebar.checkbox('Check here')==True:
    v=st.sidebar.text_input('input vertical angle')
    h=st.sidebar.text_input('input horizontal angle')

uploaded_file=st.sidebar.file_uploader("", type='png')
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    st.image(
        image, caption='uploaded images',
        use_column_width=True,
        width=1
    )
    ans,segmap=test_semseg.main(img_array,v,h)
    st.write("distance =",'{:.1f}'.format(ans*100),"cm" )
    agree = st.checkbox('Result of semantic segmentation')
    if agree == True :
        segmap = np.asarray(segmap)
        AA=np.zeros((segmap.shape[0],segmap.shape[1],3),dtype="uint8")
        AA[:,:,:]=0
        AA[:,:,1]=segmap
        AA*=200
        blend=cv2.addWeighted(img_array,0.4,AA,0.6,0)
        st.image(
            blend,
            use_column_width=True,
        )




