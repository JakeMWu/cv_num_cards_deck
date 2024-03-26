import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time 

background_image_url = "https://hamnersunbelievable.com/wp-content/uploads/2023/05/Silhouette-of-magician-with-a-wand-1200x900.jpg"
# CSS to inject contained in a multiline string
background_style = """
<style>
body {{
background-image: url(f"{background_image_url}");
background-size: cover;
}}
</style>
"""

st.markdown(background_style, unsafe_allow_html=True)

card_names = {
    '2c': '2 of Clubs', '2d': '2 of Diamonds', '2h': '2 of Hearts', '2s': '2 of Spades',
    '3c': '3 of Clubs', '3d': '3 of Diamonds', '3h': '3 of Hearts', '3s': '3 of Spades',
    '4c': '4 of Clubs', '4d': '4 of Diamonds', '4h': '4 of Hearts', '4s': '4 of Spades',
    '5c': '5 of Clubs', '5d': '5 of Diamonds', '5h': '5 of Hearts', '5s': '5 of Spades',
    '6c': '6 of Clubs', '6d': '6 of Diamonds', '6h': '6 of Hearts', '6s': '6 of Spades',
    '7c': '7 of Clubs', '7d': '7 of Diamonds', '7h': '7 of Hearts', '7s': '7 of Spades',
    '8c': '8 of Clubs', '8d': '8 of Diamonds', '8h': '8 of Hearts', '8s': '8 of Spades',
    '9c': '9 of Clubs', '9d': '9 of Diamonds', '9h': '9 of Hearts', '9s': '9 of Spades',
    '10c': '10 of Clubs', '10d': '10 of Diamonds', '10h': '10 of Hearts', '10s': '10 of Spades',
    'Jc': 'Jack of Clubs', 'Jd': 'Jack of Diamonds', 'Jh': 'Jack of Hearts', 'Js': 'Jack of Spades',
    'Qc': 'Queen of Clubs', 'Qd': 'Queen of Diamonds', 'Qh': 'Queen of Hearts', 'Qs': 'Queen of Spades',
    'Kc': 'King of Clubs', 'Kd': 'King of Diamonds', 'Kh': 'King of Hearts', 'Ks': 'King of Spades',
    'Ac': 'Ace of Clubs', 'Ad': 'Ace of Diamonds', 'Ah': 'Ace of Hearts', 'As': 'Ace of Spades',
}
# Load your TensorFlow model we want to do this from a URL as its too big for github 
MODEL_PATH = 'models/model_to_use'
model = tf.keras.models.load_model(MODEL_PATH)

st.title('2c or not 2c, that is the question')
uploaded_file = st.file_uploader("Upload the card you're thinking of...", type="jpg")

if uploaded_file is not None:
    image_1 = Image.open(uploaded_file)
    st.image(image_1, caption='Uploaded Image.', use_column_width=True)
    image = Image.open(uploaded_file).resize((224, 224)).convert('RGB')
    
    st.write("")
    st.write("The card you're thinking of...")

    # Preprocess the image to fit your model's input requirements
    image = np.array(image) / 255
    image = np.expand_dims(image, axis=0) # add a batch dimension 

    # Make a prediction
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    #st.write(predicted_class)
    class_names = ['10c', '10d', '10h', '10s', '2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s',
                    '4c', '4d', '4h', '4s', '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s',
                      '7c', '7d', '7h', '7s', '8c', '8d', '8h', '8s', '9c', '9d', '9h', '9s', 
                      'Ac', 'Ad', 'Ah', 'As', 'Jc', 'Jd', 'Jh', 'Js', 'Kc', 'Kd', 'Kh', 'Ks', 'Qc', 'Qd', 'Qh', 'Qs'] # Customize based on your classes
    #string = "This image most likely belongs to {class_names[predicted_class]} with a {:.2f} percent confidence."
    time.sleep(1) # 5 seconds for dramatic effect 
    #st.write(string.format(class_names[predicted_class], 100 * np.max(predictions)))
    
    st.write(f"Is the {card_names[class_names[predicted_class]]}!!!!")
