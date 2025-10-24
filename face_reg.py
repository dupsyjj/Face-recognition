import cv2
import streamlit as st
import numpy as np

st.markdown("<h1 style='color: #FFACAC'>FACE DETECTION APP</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='color: #F2921D'>Built by Modupe Oshinjirin</h6>", unsafe_allow_html=True)

st.image('pngwing.com (14).png', caption='Built by Modupe Oshinjirin', width=400)
st.markdown('<hr><hr><br>', unsafe_allow_html=True)

# Usage instructions
if st.button('Read the usage Instructions below'):
    st.success('Hello User, these are the guidelines for the app usage')
    st.write('Press the camera button for our model to detect your face')
    st.write('Use the MinNeighbour slider to adjust how many neighbors each candidate rectangle should have to retain it')
    st.write('Use the Scale slider to specify how much the image size is reduced at each image scale')

st.markdown('<br>', unsafe_allow_html=True)

# Load the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Sliders for parameters
min_Neighbours = st.slider('Adjust Min Neighbour', 1, 10, 5)
Scale_Factor = st.slider('Adjust Scale Factor', 1.01, 3.0, 1.3, 0.01)

# Placeholder to display frames
frame_placeholder = st.empty()

# Button to start face detection
if st.button('FACE DETECT'):
    camera = cv2.VideoCapture(0)  # open webcam
    
    while True:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to capture webcam frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=Scale_Factor, minNeighbors=min_Neighbours, minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 255, 0), 2)

        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Break loop if Streamlit stops (approximation)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
