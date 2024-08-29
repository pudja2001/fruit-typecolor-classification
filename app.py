import streamlit as st
import requests
from PIL import Image

def main():
    st.title("Fruit Classification App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # st.image(image, caption="Uploaded Image", use_column_width=True)
        st.image(image, caption="Uploaded Image", width=300) 

        if st.button("Classify"):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post("http://localhost:8000/predict", files=files)
            
            if response.status_code == 200:
                result = response.json()["prediction"]
                st.success(f"Prediction: {result}")
            else:
                st.error("Error in classification. Please try again.")

if __name__ == "__main__":
    main()