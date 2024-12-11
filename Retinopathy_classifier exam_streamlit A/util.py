# util.py
import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import tensorflow as tf

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def classify(image, model, class_names):
    """
    This function takes an OCT image, a model, and a list of class names and returns the predicted class and confidence score.
    """
    # Convert image to (128, 128)
    image = ImageOps.fit(image, (128, 128), Image.Resampling.LANCZOS)
    
    # Convert image to numpy array
    image_array = np.asarray(image)
    
    # Normalize image
    normalized_image_array = image_array.astype(np.float32) / 255.0
    
    # Set model input
    data = np.expand_dims(normalized_image_array, axis=0)
    
    # Make prediction
    prediction = model.predict(data)
    
    # Get index of the class with highest probability
    index = np.argmax(prediction[0])
    
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    return class_name, confidence_score
