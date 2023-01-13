import streamlit as st
from multiapp import MultiApp
from apps import video, image # import your app modules here

st.set_page_config(layout='wide', initial_sidebar_state='collapsed')
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

app = MultiApp()

st.sidebar.title("Image Processing App")

st.markdown("""
# Image Processing and Cell Counting App

This is a simple application that can process your images and videos.  \n
Contents:  \n
    1. Image Processing and Cell Counting  \n
    2. Video Processing and Cell Counting

""")

# Add all your application here
app.add_app("Image", image.app)
app.add_app("Video", video.app)

# The main app
app.run()
