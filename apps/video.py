import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
import moviepy.editor as mv
import pandas as pd
from io import BytesIO
import plost
from os import path
from .modules import vid_count, to_excel_c, vid_quality

def app():
    # Page Title
    st.header('Video Processing Page')

    # Sidebar 
    st.sidebar.subheader('Upload Video')
    uploaded_video = st.sidebar.file_uploader(
        'Choose your video file.', 
        type=['mp4', 'avi', 'mov'], 
        accept_multiple_files=False)
    
    if uploaded_video is not None:
        # Video Processing
        df_counts, temp_file_name = vid_count(uploaded_video=uploaded_video)
        avi = mv.VideoFileClip('processed_video.avi')
        avi.write_videofile('processed_video.mp4')
        processed_mp4 = open('processed_video.mp4', 'rb')
        video_bytes = processed_mp4.read()

        # Video Quality
        vid_quality_df = vid_quality(tempfile_name=temp_file_name)

        # Mean Video Quality Metrics
        mean_signal_to_noise = np.mean(vid_quality_df['SNR'])
        mean_signal_to_noise = round(mean_signal_to_noise, 2)
        mean_contrast_ratio = np.mean(vid_quality_df['CR'])
        mean_contrast_ratio = round(mean_contrast_ratio, 2)
        mean_min_centroid_intesity = np.mean(vid_quality_df['Min. CI'])
        mean_min_centroid_intesity = round(mean_min_centroid_intesity, 2)
        mean_max_centroid_intensity = np.mean(vid_quality_df['Max. CI'])
        mean_max_centroid_intensity = round(mean_max_centroid_intensity, 2)
        mean_centroid_intensities_std = np.mean(vid_quality_df['CI Std'])
        mean_centroid_intensities_std = round(mean_centroid_intensities_std, 2)
        mean_cut_off_value = np.mean(vid_quality_df['Cut-Off Value'])
        mean_cut_off_value = round(mean_cut_off_value, 2)

        q_col1, q_col2, q_col3, q_col4, q_col5, q_col6 = st.columns(6)

        q_col1.metric('Mean SNR', mean_signal_to_noise)
        q_col2.metric('Mean CR', mean_contrast_ratio)
        q_col3.metric('Mean Min. CI', mean_min_centroid_intesity)
        q_col4.metric('Mean Max. CI', mean_max_centroid_intensity)
        q_col5.metric('Mean CI STD', mean_centroid_intensities_std)
        q_col6.metric('Mean Cut-Off Value', mean_cut_off_value)


        vid_col1, vid_col2 = st.columns(2)
        with vid_col1:
            # Raw Video
            st.subheader('Raw Video')
            st.video(uploaded_video)

            # Counts DF
            df_counts_xlsx = to_excel_c(df=df_counts)
            st.subheader('Counts per Frame')
            st.dataframe(df_counts, use_container_width=True)
            st.download_button(
                        label='Download',
                        data=df_counts_xlsx,
                        file_name='Counts_per_frame.xlsx'
                        )

        with vid_col2:
            # Processed Video
            st.subheader('Processed Video')
            st.video(video_bytes)

            # Frame Quelities DF
            vid_quality_df_xlsx = to_excel_c(df=vid_quality_df)
            st.subheader('Frame Quality Metrics')
            st.dataframe(vid_quality_df, use_container_width=True)
            st.download_button(
                        label='Download',
                        data=vid_quality_df_xlsx,
                        file_name='Counts_per_frame.xlsx'
                        )

        