# main.py
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
from util import classify, set_background

# Set background
set_background('C:/startercoding/Retinopathy_classifier exam_streamlit A/bg1.jpg')

# Set title
st.title('OCT Image Classification')

# Add some space
st.text_input("", label_visibility="hidden")

# Set header
st.header('Please upload an OCT image')

# Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load classifier
model = load_model('C:/startercoding/Retinopathy_classifier exam_streamlit A/OCT_CNN_model.keras')

# Load class names
class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# Display image and classification result
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)
    
    # Classify image
    class_name, conf_score = classify(image, model, class_names)
    
    # Write classification
    st.write("## {}".format(class_name))
    st.write("### Confidence: {}%".format(int(conf_score * 1000) / 10))

