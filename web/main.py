import streamlit as st
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt



def upload_image():
    uploaded_image = st.file_uploader("Choose an image", accept_multiple_files=False)

    if uploaded_image:
        st.write('Your image: ')
        st.image(uploaded_image,caption=uploaded_image.name)

        uploaded_image = Image.open(uploaded_image)
        uploaded_image = np.array(uploaded_image)
    
    return uploaded_image


if __name__ == '__main__':
    # Title
    st.title('Feet size estimation')

    img = upload_image()
    if img is not None:
        is_process = st.button("Let's find your feet size")
        if is_process:
            st.write('dep trai co gi sai')
        else:
            st.write('co cai dau bui')