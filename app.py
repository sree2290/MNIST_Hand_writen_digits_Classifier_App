import streamlit as st
import numpy as np

from PIL import Image
from pickle import load



knn_classifier = load(open('models/knn_model.pkl', 'rb'))

st.header("MNIST Hand Writen Digits Classifier")
image_file = st.file_uploader("Upload Numaric Image", type=["png","jpg","jpeg"])



#uploaded_file = st.file_uploader("Choose a file")
if image_file is not None:
     # To read file as bytes:
     st.image(image_file, width=250)
     photos = Image.open(image_file)
     photos_gray_0 = photos.convert("L")
     photos_gray_0 = photos_gray_0.resize((28,28))
     photos_arr_0 = np.array(photos_gray_0)

     btn_click = st.button('Predict')
     if btn_click == True:
        prediction = knn_classifier.predict([photos_arr_0.ravel()])
        st.title(f'The Number Predicted is  : {prediction[0]}')
st.text("Made with Love ❤")
        #st.title(prediction[0])
