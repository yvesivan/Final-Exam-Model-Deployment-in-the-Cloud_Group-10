import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import io

st.set_page_config(page_title="Weather Detection System", page_icon=":partly_sunny:")

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('Weather_Classification-Model.h5')
    return model

model = load_model()

st.title("Weather Detection System")
st.write("Cloudy: 🌥️  Rainy: 🌧️  Shiny: ☀️  Sunrise: 🌅")

file = st.file_uploader("Upload a weather photo", type=["jpg", "jpeg", "png", "heic"])

if file is None:
    st.write("Please upload an image file")
else:
    # Handling HEIC format
    if file.type == 'application/octet-stream':
        st.write("Converting HEIC to JPEG format...")
        content = file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    else:
        image = Image.open(file)

    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Click here to detect the weather'):
        st.write("Detecting... 🔄")
        size = (150, 150)  
        image = ImageOps.fit(image, size)
        img = np.asarray(image)
        
        # Normalize image array to [0, 1]
        img = img / 255.0
        
        # Add a batch dimension
        img_reshape = np.expand_dims(img, axis=0)
        
        prediction = model.predict(img_reshape)
        
        class_names = ['🌥️ Cloudy', '🌧️ Rainy', '☀️ Shiny', '🌅 Sunrise']
        st.write(f"Predicted weather: {class_names[np.argmax(prediction)]}")
