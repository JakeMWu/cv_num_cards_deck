import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests

def download_model(url, save_path):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(r.content)
        print("Model downloaded successfully.")
    else:
        print("Failed to download the model.")

# Load your TensorFlow model we want to do this from a URL as its too big for github 
MODEL_PATH = 'models/model_3'
model = tf.keras.models.load_model(MODEL_PATH)

st.title('Image Classification App')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224)).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image to fit your model's input requirements
    image = np.array(image)
    # Example preprocessing, adjust depending on your model
    image = np.array(image) / 127.5 - 1
    image = np.expand_dims(image, axis=0)

    # Make a prediction
    predictions = model.predict(image)
    class_names = ['10c', '10d', '10h', '10s', '2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s', '4c', '4d', '4h', '4s', '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s', '7c', '7d', '7h', '7s', '8c', '8d', '8h', '8s', '9c', '9d', '9h', '9s', 'Ac', 'Ad', 'Ah', 'As', 'Jc', 'Jd', 'Jh', 'Js', 'Kc', 'Kd', 'Kh', 'Ks', 'Qc', 'Qd', 'Qh', 'Qs'] # Customize based on your classes
    string = "This image most likely belongs to {} with a {:.2f} percent confidence."
    st.write(string.format(class_names[np.argmax(predictions)], 100 * np.max(predictions)))