import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Melanoma Detection",
    page_icon="🩺",
    layout="centered"
)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("melanoma_vgg16_model.h5")
    return model

model = load_model()

IMG_SIZE = 224

def preprocess_image(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)

    if len(image.shape) == 2:
        image = np.stack([image]*3, axis=-1)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    return image

st.title("🩺 Melanoma Detection System")

st.write(
"""
Upload a dermoscopic or clinical skin lesion image.

The AI model predicts whether the lesion is:

- Benign (Non-Melanoma)
- Melanoma

The application also provides a confidence score.
"""
)

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg","jpeg","png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    input_image = preprocess_image(image)

    prediction = model.predict(input_image)[0][0]

    melanoma_probability = float(prediction)
    benign_probability = 1 - melanoma_probability

    threshold = 0.5

    st.write("---")

    if melanoma_probability >= threshold:

        confidence = melanoma_probability * 100

        st.error("### Prediction: Melanoma")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.warning(
        """
        **Recommendation**

        • Consult a dermatologist immediately.

        • Do not rely solely on AI predictions.

        • A biopsy and clinical examination are required for confirmation.

        • Avoid delaying medical consultation.
        """
        )

    else:

        confidence = benign_probability * 100

        st.success("### Prediction: Benign (Non-Melanoma)")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.info(
        """
        **Recommendation**

        • Continue regular skin self-examinations.

        • Monitor for changes using the ABCDE rule.

        • Seek medical advice if the lesion changes in size, color, or shape.
        """
        )

    st.write("---")

    st.subheader("Prediction Probabilities")

    st.write(f"Melanoma : **{melanoma_probability:.4f}**")

    st.write(f"Benign : **{benign_probability:.4f}**")
