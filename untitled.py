import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your TensorFlow model
MODEL_PATH = 'path/to/your/saved_model'
model = tf.keras.models.load_model(MODEL_PATH)

st.title('Image Classification App')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image to fit your model's input requirements
    image = np.array(image)
    # Example preprocessing, adjust depending on your model
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Make a prediction
    predictions = model.predict(image)
    class_names = ['class1', 'class2', 'class3'] # Customize based on your classes
    string = "This image most likely belongs to {} with a {:.2f} percent confidence."
    st.write(string.format(class_names[np.argmax(predictions)], 100 * np.max(predictions)))