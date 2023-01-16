import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
import moviepy.editor as mv
from .modules import img_count, img_quality

def app():
    # Page Title
    st.header('Image Processing Page')

    # Sidebar 
    st.sidebar.subheader('Upload Image')
    uploaded_img = st.sidebar.file_uploader(
        'Choose your image file or files.', 
        type=['png' , 'jpg', 'jpeg', 'tif'], 
        accept_multiple_files=False)
    
    if uploaded_img is not None:
        # Counts
        lines, top_left_count, top_right_count, bottom_right_count, bottom_left_count, temp_file_name = img_count(uploaded_img=uploaded_img)

        # Image Quality
        signal_to_noise, contrast_ratio, min_centroid_intesity, max_centroid_intensity, centroid_intensities_std, cut_off_value = img_quality(tempfile_name=temp_file_name)
        
        st.write(lines.shape)
        # ROW 1 - Quality Metrics
        q_col1, q_col2, q_col3, q_col4, q_col5, q_col6 = st.columns(6)

        q_col1.metric('SNR', signal_to_noise)
        q_col2.metric('CR', contrast_ratio)
        q_col3.metric('Min. CI', min_centroid_intesity)
        q_col4.metric('Max. CI', max_centroid_intensity)
        q_col5.metric('CI STD', centroid_intensities_std)
        q_col6.metric('Cut-Off Value', cut_off_value)
        
        #ROW 2 - Images
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            # Raw Image
            st.subheader('Raw Image')
            st.image(uploaded_img)
        
        with img_col2:
            # Processed Image
            st.subheader('Processed Image')
            st.image(lines)

        # ROW 3 - Count Metrics
        st.subheader('Counts')
        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric('Top Left Count', top_left_count)
        col2.metric('Top Right Count', top_right_count)
        col3.metric('Bottom Left Count', bottom_left_count)
        col4.metric('Bottom Right Count', bottom_right_count)
        all_counts = top_left_count + top_right_count + bottom_right_count + bottom_left_count
        col5.metric('All Counts', all_counts)
        
