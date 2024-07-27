import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the Keras model
model = load_model('detection.h5')

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((100, 100))  # Resize image to match model's expected sizing
    img = np.array(img)  # Convert image to numpy array
    if len(img.shape) == 2:  # If the image is grayscale (only height and width)
        img = np.stack([img] * 3, axis=-1)  # Convert grayscale to RGB by stacking the single channel
    img = img / 255.0  # Normalize pixel values to range 0 to 1
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Class labels
class_label = {0: 'No Tumor', 1: 'Tumor'}

# Streamlit app
st.title('Brain Tumor Detection')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Predict
    prediction = model.predict(processed_image)
    predicted_class = class_label[int(np.round(prediction)[0][0])]

    st.write('Prediction:', predicted_class)
    st.write('Confidence:', round(prediction[0][0] * 100, 2), '%')
