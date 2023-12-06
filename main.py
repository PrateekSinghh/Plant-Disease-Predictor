import streamlit as st
from PIL import Image
import numpy as np
import model_loader as ml
import pandas as pd

st.set_page_config(
    page_title="Plant Disease Predictor",
    page_icon="favicon.png"
)
st.markdown("# Plant Disease Predictor", unsafe_allow_html=True)
st.markdown("---")

plant_type = st.sidebar.selectbox(
    label="Choose plant type",
    options=ml.get_plants()
)

plant_image = st.sidebar.file_uploader(
    label="Upload your image",
    type=["png", "jpg", "jpeg"]
)


def process_and_display_image(image): 
    img = Image.open(image)
    img.thumbnail((800, 500))  # Keep the aspect ratio intact
    return img


placeholder = st.empty()

if plant_image and "image" in plant_image.type:
    st.image(
        process_and_display_image(image=plant_image),
        channels="RGB"
    )


def read_image(data) -> np.ndarray:
    input_image = Image.open(data)
    resized_image = input_image.resize((256, 256))
    final_image = resized_image.convert("RGB")
    return np.array(final_image)


def predict_disease(plant, image) -> pd.DataFrame:
    model = ml.load_model(plant=plant_type)
    plant_diseases = ml.get_disease(plant=plant)

    img = read_image(image)
    img_batch = np.expand_dims(img, 0)

    prediction = np.array([x * 100 for x in model.predict(img_batch)[0]])
    df = pd.DataFrame(
        data=prediction.reshape(1, -1), columns=plant_diseases, index=["Confidence %"]
    )

    return df


placeholder = st.empty()

if st.button(label="Run Prediction") and plant_image:
    placeholder.write("Please wait...")
    data = predict_disease(plant=plant_type, image=plant_image)
    placeholder.empty()
    st.dataframe(data=data)
