import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

@st.cache_data(experimental_allow_widgets=True)
def load_model():
    model = tf.keras.models.load_model('Weather_Classification-Model.h5')
    return model

model = load_model()

st.write("""
# Weather Detection System
""")

file = st.file_uploader("Choose weather photo from computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (150, 150)  
    image = ImageOps.fit(image_data, size)
    img = np.asarray(image)
    
    # Normalize image array to [0, 1]
    img = img / 255.0
    
    # Add a batch dimension
    img_reshape = np.expand_dims(img, axis=0)

    st.write(f"Image shape: {img_reshape.shape}")
    st.write(f"Image dtype: {img_reshape.dtype}")
    
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)
