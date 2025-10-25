import cv2
import streamlit as st
import numpy as np

# App title and description
st.markdown("<h1 style='color:#FFACAC'>Face Detection App</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='color:#F2921D'>Built by Modupe Oshinjirin</h6>", unsafe_allow_html=True)

# Display image
st.image('pngwing.com (14).png', caption='Built by Modupe Oshinjirin', width=400)
st.markdown('<hr><br>', unsafe_allow_html=True)

# Instructions
if st.button('Read Usage Instructions'):
    st.success("Guidelines for using the app:")
    st.write("- Press 'Start Detection' to activate your webcam.")
    st.write("- Use sliders to adjust Min Neighbors and Scale Factor.")
    st.write("- Pick a color for rectangles drawn around faces.")

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# User controls
min_neighbors = st.slider('Min Neighbors', 1, 10, 5)
scale_factor = st.slider('Scale Factor', 1.01, 3.0, 1.3, 0.01)
rect_color = st.color_picker('Pick rectangle color', '#FF0000')
rect_color_bgr = tuple(int(rect_color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))

# Placeholder for webcam frames
frame_placeholder = st.empty()

# Start detection
if st.button('Start Detection'):
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to capture webcam frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color_bgr, 2)

        # Convert to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
    
    camera.release()
