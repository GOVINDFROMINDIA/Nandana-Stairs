import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

np.set_printoptions(suppress=True)

model = load_model("keras_Model.h5", compile=False)

class_names = open("labels.txt", "r").readlines()

def predict_image(image):
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score

def main():
    st.title("Nandana Stairs")

    open_camera_checkbox = st.checkbox("Open Camera")
    if open_camera_checkbox:
        st.sidebar.write("Opening Camera...")
        camera = cv2.VideoCapture(0)
        while open_camera_checkbox:
            ret, image = camera.read()
            class_name, confidence_score = predict_image(image)
            st.image(image, caption=f"Class: {class_name[2:]}, Confidence: {np.round(confidence_score * 100)}%",
                     use_column_width=True)
            open_camera_checkbox = st.checkbox("Stop Camera")
            st.markdown(f"<p style='font-size:30px; color:#000000; font-weight:bold;'>Class: {class_name[2:]}</p>",
                        unsafe_allow_html=True)

    select_image_checkbox = st.checkbox("Select Image")
    if select_image_checkbox:
        st.sidebar.write("Choose a file")
        uploaded_file = st.sidebar.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.sidebar.write("File Selected!")
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            class_name, confidence_score = predict_image(image)
            st.image(image, caption=f"Class: {class_name[2:]}, Confidence: {np.round(confidence_score * 100)}%",
                     use_column_width=True)
            st.markdown(f"<p style='font-size:30px; color:#000000; font-weight:bold;'>Class: {class_name[2:]}</p>",
                        unsafe_allow_html=True)

if __name__ == "__main__":
    main()
