
import streamlit as st
from PIL import Image

from predict import load_model, predict_image


# --------------------------------
# 1. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🧠",
    layout="centered"
)


# --------------------------------
# 2. Application Title
# --------------------------------

st.title("CIFAR-10 Image Classification")

st.write(
    "Upload an image and let the CNN Version 2 model "
    "predict its class."
)

st.caption(
    "Classes: Airplane, Automobile, Bird, Cat, Deer, "
    "Dog, Frog, Horse, Ship and Truck."
)


# --------------------------------
# 3. Load Model
# --------------------------------

@st.cache_resource
def get_model():

    return load_model()


try:

    model = get_model()

    st.success("Model loaded successfully.")

except Exception as e:

    st.error(f"Model loading failed: {e}")

    st.stop()


# --------------------------------
# 4. Image Upload
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------
# 5. Image Prediction
# --------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Input Image",
        width=300
    )

    if st.button(
        "Predict Image",
        type="primary",
        use_container_width=True
    ):

        try:

            result = predict_image(
                uploaded_file,
                model
            )

            st.divider()

            st.subheader("Prediction Results")

            predicted_class = result["predicted_class"]

            confidence = result["confidence"]

            st.markdown(
                f"### Predicted Class: {predicted_class.title()}"
            )

            st.metric(
                label="Prediction Confidence",
                value=f"{confidence:.2f}%"
            )

            st.progress(
                min(max(confidence / 100, 0.0), 1.0)
            )

            if confidence < 80:

                st.warning(
                    "Low confidence prediction. "
                    "The model may be uncertain."
                )

            else:

                st.success(
                    "Prediction confidence "
                )

            # Top 3 Predictions

            st.subheader("Top 3 Predictions")

            for index, item in enumerate(
                result["top_3"],
                start=1
            ):

                st.write(
                    f"{index}. {item['class'].title()} "
                    f"- {item['confidence']:.2f}%"
                )

                st.progress(
                    min(
                        max(item["confidence"] / 100, 0.0),
                        1.0
                    )
                )

        except Exception as e:

            st.error(f"Prediction failed: {e}")


# --------------------------------
# 6. Footer
# --------------------------------

st.divider()

st.caption(
    "CIFAR-10 Image Classification | "
    "PyTorch | CNN Version 2"
)
