from cgi import test
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import skimage 
from utils import *
import os
import shutil
from classifier import *
import pipeline

def upload_image():
    uploaded_image = st.file_uploader("Choose an image", accept_multiple_files=False)

    if uploaded_image:
        st.write('Your image: ')
        st.image(uploaded_image,caption=uploaded_image.name,use_column_width=True)

        uploaded_image = Image.open(uploaded_image)
        uploaded_image = np.array(uploaded_image,np.uint8)
    
    return uploaded_image


# Title
st.title('Feet size estimation')
fh, fw = 0, 0
st.write('Please upload your feet image to estimate')
img = upload_image()
if img is not None:
    is_process = st.button("Let's find your feet size")
    if is_process:
        wait = st.write('Your image is processing, Please wait!')
        #prepocess
        img_class = classifier(img)
        preprocess_img = preprocess(img, img_class)
        if preprocess_img is not None:
            del wait
        # segments of foot, paper, paper's corners...

        clustered_img = kMeans_cluster(preprocess_img)
        
        st.title("Here is the image after clustering")
        st.image(clustered_img, caption='Clustered Image', use_column_width=True)

        edge_detected_img = paperEdgeDetection(clustered_img)
        
        st.title("Here is the image after edge detection")
        st.image(edge_detected_img, caption='Edge Detected Image', use_column_width=True)

        boundRect, contours, contours_poly, img = getBoundingBox(edge_detected_img)
        
        st.title("Here is the image after bounding box")
        st.image(img, caption='Bounding Box Image', use_column_width=True)
        
        if img_class == 0:
            cropped_img, pcropped_img = cropOrig(boundRect[0], clustered_img)
        else:
            cropped_img, pcropped_img = cropOrig(boundRect[1], clustered_img)
            
        st.title("Here is the image after cropping")
        st.image(cropped_img, caption='Cropped Image', use_column_width=True)

        new_img = overlayImage(cropped_img, pcropped_img)
        
        st.title("Here is the image after overlay")
        st.image(new_img, caption='Overlay Image', use_column_width=True)

        fedged = footEdgeDetection(new_img)
        fboundRect, fcnt, fcntpoly, fimg = getBoundingBox(fedged)
        fdraw = drawCnt(fboundRect[0], fcnt, fcntpoly, fimg)
        
        st.title("Here is the image after edge detection")
        st.image(fdraw, caption='Edge Detected Image', use_column_width=True)

        ofs, fh, fw, ph, pw = calcFeetSize(pcropped_img, fboundRect)
        
        st.title(f"[INFO] Feet size (cm): {ofs}")
        st.write(f"[INFO] Width (cm): {fw}")
        st.write(f"[INFO] Height(cm): {fh}")
    