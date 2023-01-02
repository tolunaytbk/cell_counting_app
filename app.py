import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
import moviepy.editor as mv

# Side Bar
st.sidebar.title('Cell Counting App')

# Video Uploader
st.sidebar.subheader('Upload Video')
uploaded_video = st.sidebar.file_uploader('Choose your video file.', type=['mp4'])

# İmage Uploader
st.sidebar.subheader('Upload Image')
uploaded_img = st.sidebar.file_uploader(
    'Choose your image file or files.', 
    type=['jpeg', 'jpg', 'png'], 
    accept_multiple_files=False)

# Main
st.title('Cell Counting Application')

# Definitions
def euc_dist(cx1, cx2, cy1, cy2):
    return np.sqrt(np.square(cx2 - cx1) + np.square(cy2 - cy1))

def img_count(uploaded_img):
    # Temp File
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_img.read())

    # Read Image from Temp File
    raw_img = cv2.imread(temp_file.name, cv2.IMREAD_UNCHANGED)

    img_height = raw_img.shape[0]
    img_width = raw_img.shape[1]

    part_num_w = 2
    part_num_h = 2

    width_first_coord = img_width / part_num_w
    width_first_coord = round(width_first_coord)

    height_first_coord = img_height / part_num_h
    height_first_coord = round(height_first_coord)

    # Gray Scale Image
    image_grayscale = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

    # Blur Image
    image_blurred = cv2.GaussianBlur(image_grayscale, (11,11), 0)

    # Canny
    image_canny = cv2.Canny(image_blurred, 0, 132, 5)

    # Dilation
    image_dilated = cv2.dilate(image_canny, (1,1), iterations = 4)

    # Contours
    (cnt, heirarchy) = cv2.findContours(image_dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Draw Contours
    result = raw_img
    cv2.drawContours(result, cnt, -1, (255, 0, 0), 1)

    # Centroids
    top_left_count = 0
    top_right_count = 0
    bottom_left_count = 0
    bottom_right_count = 0
    centroid_coords = []
    for c in cnt:
        M = cv2.moments(c)
        if int(M['m10']) != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        
        threshold = 25
        if (cx, cy) not in centroid_coords and centroid_coords:
            centroid_coords_dist = []
            for i in range(len(centroid_coords)):
                dist = euc_dist(cx, centroid_coords[i][0], cy, centroid_coords[i][1])
                centroid_coords_dist.append(int(dist))
            
            smallest_dist = min(centroid_coords_dist)
            print(smallest_dist)
            if smallest_dist >= threshold:
                centroid_coords.append((cx, cy))

                # Counted Cells
                if cx < width_first_coord and cy < height_first_coord:
                    top_left_count += 1
                if cx > width_first_coord and cy < height_first_coord:
                    top_right_count += 1
                if cx < width_first_coord and cy > height_first_coord:
                    bottom_left_count += 1
                if cx > width_first_coord and cy > height_first_coord:
                    bottom_right_count += 1
        else:
            centroid_coords.append((cx, cy))

            # Counted Cells
            if cx < width_first_coord and cy < height_first_coord:
                top_left_count += 1
            if cx > width_first_coord and cy < height_first_coord:
                top_right_count += 1
            if cx < width_first_coord and cy > height_first_coord:
                bottom_left_count += 1
            if cx > width_first_coord and cy > height_first_coord:
                bottom_right_count += 1

    for i in range(len(centroid_coords)):
        cv2.circle(result, centroid_coords[i], 2, (0, 0, 0), -1)

    # Result
    lines = cv2.line(result, (width_first_coord, 0), (width_first_coord, img_height), (0, 0, 255))
    lines = cv2.line(lines, (0, height_first_coord), (img_width, height_first_coord), (0, 0, 255))

    # Texts
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    # Top-Left Part Texts
    lines = cv2.putText(lines, f"Top-Left Counted Cells: {top_left_count}", (10, 10), font, 0.5, (50, 255, 50), 1, cv2.LINE_AA)

    # Top-Right Part Texts
    lines = cv2.putText(lines, f"Top-Right Counted Cells: {top_right_count}", (width_first_coord + 10, 10), font, 0.5, (50, 255, 50), 1, cv2.LINE_AA)

    # Bottom-Left Part Texts
    lines = cv2.putText(lines, f"Bottom-Left Counted Cells: {bottom_left_count}", (10, height_first_coord + 10), font, 0.5, (50, 255, 50), 1, cv2.LINE_AA)

    # Bottom-Right Part Texts
    lines = cv2.putText(lines, f"Bottom-Right Counted Cells: {bottom_right_count}", (width_first_coord + 10, height_first_coord + 10), font, 0.5, (50, 255, 50), 1, cv2.LINE_AA)

    # All Counts Text
    all_counts = top_left_count + top_right_count + bottom_left_count + bottom_right_count
    lines = cv2.putText(lines, f"All Counted Cells: {all_counts}", (10, img_height - 10), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    lines = cv2.cvtColor(lines, cv2.COLOR_BGR2RGB)
    return lines, top_left_count, top_right_count, bottom_right_count, bottom_left_count

def vid_count(uploaded_video):
    # Temp File 
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_video.read())

    cap = cv2.VideoCapture(temp_file.name)

    top_left_counts = []
    top_right_counts = []
    bottom_left_counts = []
    bottom_right_counts = []
    total_counts = []
    size = (640, 480)
    new_video = cv2.VideoWriter('processed_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps=30, frameSize=size)
    while(cap.isOpened()):
        ret, frame = cap.read()

        raw_img = frame
        
        if raw_img is None:
            break

        img_height = raw_img.shape[0]
        img_width = raw_img.shape[1]

        part_num_w = 2
        part_num_h = 2

        width_first_coord = img_width / part_num_w
        width_first_coord = round(width_first_coord)

        height_first_coord = img_height / part_num_h
        height_first_coord = round(height_first_coord)

        # Gray Scale Image
        image_grayscale = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

        # Blur Image
        image_blurred = cv2.GaussianBlur(image_grayscale, (11,11), 0)

        # Canny
        image_canny = cv2.Canny(image_blurred, 50, 132, 5)

        # Dilation
        image_dilated = cv2.dilate(image_canny, (1,1), iterations = 4)

        # Contours
        (cnt, heirarchy) = cv2.findContours(image_dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Draw Contours
        result = raw_img
        cv2.drawContours(result, cnt, -1, (255, 0, 0), 1)

        # Centroids
        top_left_count = 0
        top_right_count = 0
        bottom_left_count = 0
        bottom_right_count = 0
        centroid_coords = []
        for c in cnt:
            M = cv2.moments(c)
            if int(M['m10']) != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            
            threshold = 25
            if (cx, cy) not in centroid_coords and centroid_coords:
                centroid_coords_dist = []
                for i in range(len(centroid_coords)):
                    dist = euc_dist(cx, centroid_coords[i][0], cy, centroid_coords[i][1])
                    centroid_coords_dist.append(int(dist))
                
                smallest_dist = min(centroid_coords_dist)

                if smallest_dist >= threshold:
                    centroid_coords.append((cx, cy))

                    # Counted Cells
                    if cx < width_first_coord and cy < height_first_coord:
                        top_left_count += 1
                    if cx > width_first_coord and cy < height_first_coord:
                        top_right_count += 1
                    if cx < width_first_coord and cy > height_first_coord:
                        bottom_left_count += 1
                    if cx > width_first_coord and cy > height_first_coord:
                        bottom_right_count += 1
            else:
                centroid_coords.append((cx, cy))

                # Counted Cells
                if cx < width_first_coord and cy < height_first_coord:
                    top_left_count += 1
                if cx > width_first_coord and cy < height_first_coord:
                    top_right_count += 1
                if cx < width_first_coord and cy > height_first_coord:
                    bottom_left_count += 1
                if cx > width_first_coord and cy > height_first_coord:
                    bottom_right_count += 1

        for i in range(len(centroid_coords)):
            cv2.circle(result, centroid_coords[i], 2, (0, 0, 0), -1)
        # Result
        lines = cv2.line(result, (width_first_coord, 0), (width_first_coord, img_height), (0, 0, 255))
        lines = cv2.line(lines, (0, height_first_coord), (img_width, height_first_coord), (0, 0, 255))

        # Texts
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL

        # Top-Left Part Texts
        lines = cv2.putText(lines, f"Top-Left Counted Cells: {top_left_count}", (10, 10), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Top-Right Part Texts
        lines = cv2.putText(lines, f"Top-Right Counted Cells: {top_right_count}", (width_first_coord + 10, 10), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Bottom-Left Part Texts
        lines = cv2.putText(lines, f"Bottom-Left Counted Cells: {bottom_left_count}", (10, height_first_coord + 10), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Bottom-Right Part Texts
        lines = cv2.putText(lines, f"Bottom-Right Counted Cells: {bottom_right_count}", (width_first_coord + 10, height_first_coord + 10), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # All Counts Text
        total_count = top_left_count + top_right_count + bottom_right_count + bottom_left_count 
        lines = cv2.putText(lines, f"All Counted Cells: {total_count}", (10, img_height - 10), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Append Count
        top_left_counts.append(top_left_count)
        top_right_counts.append(top_right_count)
        bottom_right_counts.append(bottom_right_count)
        bottom_left_counts.append(bottom_left_count)
        total_counts.append(total_count)

        #Display
        #cv2.imshow('frame', lines)

        #if cv2.waitKey(150) & 0xFF == ord('q'):
            #break
        
        rgb_lines = cv2.cvtColor(lines, cv2.COLOR_BGR2RGB)
        new_video.write(rgb_lines)
        

    # DataFrame of Counted Numbers per Frame 
    df_counts = pd.DataFrame({
        'Top-Left': top_left_counts,
        'Top-Right': top_right_counts,
        'Bottom-Right': bottom_right_counts,
        'Bottom-Left': bottom_left_counts,
        'All Counted Cells': total_counts
    })
#    df_counts.to_excel('counts/counts_per_frame.xlsx', index=False)

    cap.release()
    new_video.release()
    #cv2.destroyAllWindows()
    return df_counts

# Image Processing
if uploaded_img is not None:
    # Raw Image
    st.subheader('Raw Image')
    st.image(uploaded_img)
    img_process_init = st.button('Process the Image')

    if img_process_init:
        # Process
        lines, top_left_count, top_right_count, bottom_right_count, bottom_left_count = img_count(uploaded_img=uploaded_img)

        # ROW 1 - Count Metrics
        st.subheader('Counts')
        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric('Top Left Count', top_left_count)
        col2.metric('Top Right Count', top_right_count)
        col3.metric('Bottom Left Count', bottom_left_count)
        col4.metric('Bottom Right Count', bottom_right_count)
        all_counts = top_left_count + top_right_count + bottom_right_count + bottom_left_count
        col5.metric('All Counts', all_counts)

        # ROW 2 - Processed Image
        st.subheader('Processed Image')
        st.image(lines)

# Video Processing
if uploaded_video is not None:
    # Raw Video
    st.subheader('Raw Video')
    st.video(uploaded_video)
    vid_process_init = st.button('Process the Video')

    if vid_process_init:
        df_counts = vid_count(uploaded_video=uploaded_video)
        avi = mv.VideoFileClip('processed_video.avi')
        avi.write_videofile('processed_video.mp4')
        processed_mp4 = open('processed_video.mp4', 'rb')
        video_bytes = processed_mp4.read()

        st.subheader('Processed Video')
        st.video(video_bytes)

        st.subheader('Counts per Frame')
        st.dataframe(df_counts, use_container_width=True)
