import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------------------------------------------
# Config
# ---------------------------------------------------
MODEL_PATH = "brain_tumor_model.keras"
CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠")

# ---------------------------------------------------
# Load model (cached so it only loads once per session)
# ---------------------------------------------------
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

model = get_model()

# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("🧠 Brain Tumor Detection")
st.write("Upload a brain MRI scan and the model will classify it as "
         "Glioma, Meningioma, Pituitary, or No Tumor.")

uploaded_file = st.file_uploader("Upload MRI", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    # Preprocess to match training pipeline
    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            prediction = model.predict(img_array)
            predicted_class = CLASSES[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

        st.success(f"Prediction: **{predicted_class}**")
        st.write(f"Confidence: {confidence:.2f}%")

        # Optional: show probabilities for all classes
        st.subheader("Class probabilities")
        for cls, prob in zip(CLASSES, prediction[0]):
            st.write(f"{cls}: {prob*100:.2f}%")
else:
    st.info("Please upload an MRI image to get a prediction.")
