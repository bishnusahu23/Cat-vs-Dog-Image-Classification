import streamlit as st
import pandas as pd
import cv2
import numpy as np
import pickle



st.title("Cat & Dog Classifier")

# Sidebar for image upload
st.sidebar.header("Upload an Image")
file_upload = st.sidebar.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

# Load model
MODEL_PATH = r'/workspaces/Cat-vs-Dog-Image-Classification/classfier_model'

try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.sidebar.error(f"Model loading error: {e}")

# Display uploaded image
if file_upload:
    image = Image.open(file_upload)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = np.array(image)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict button
    if st.sidebar.button("Predict"):
        with st.spinner("⏳ Making Prediction..."):
            prediction_prob = model.predict(img)[0][0]  
            prediction = "Dog" if prediction_prob > 0.5 else "Cat"
            confidence = round(prediction_prob * 100, 2) if prediction_prob > 0.5 else round((1 - prediction_prob) * 100, 2)

        # Display results
        st.subheader(f"Model Prediction: **{prediction}**")
        st.success(f"Confidence: {confidence}%")

        if prediction == "Dog":

            st.info("Woof woof! Looks like a dog!")
        else:

            st.info("Meow! That's a cat!")

st.markdown("---")
st.markdown("👨‍💻 Developed by Bishnu sahu| Powered by Streamlit")
