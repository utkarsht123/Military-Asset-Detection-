import streamlit as st
import requests
from PIL import Image
from interactive_viz import create_confidence_chart
import io

st.title('Military Asset Detector')
st.write('Upload an image to classify (Tank, Helicopter, or Ship).')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Classifying...'):
        API_URL = "http://127.0.0.1:8000/predict"
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        try:
            response = requests.post(API_URL, files=files)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: {result['prediction']}")
                fig = create_confidence_chart(result['confidences'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Server error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to the inference server: {e}") 