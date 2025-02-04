import streamlit as st
import requests
from PIL import Image
import io
import os

# FastAPI Endpoints
PORT = os.getenv("PORT", "8000")
API_URL = "https://enviroclass-605755972351.europe-west1.run.app"
WILDFIRE_API_URL = f"{API_URL}/predict-wildfire"
ENVIRONMENT_API_URL = f"{API_URL}/predict-environment"


# Setting Page Config
st.set_page_config(page_title="🔥 Wildfire Detection Service",
                   page_icon="🔥",
                   layout="centered")

# Title & Subtitle
st.title("Wildfire Detection Service")

st.markdown("""
### 🔥 WELCOME TO YOUR WILDFIRE & AREA DETECTION SERVICE 🔥


#### Stay Alert. Stay Safe. Protect Our Planet. 🌍
Are you concerned about wildfires? Do you want to **quickly analyze satellite images** to detect potential wildfire hazards?

This service allows you to **upload satellite images**, and our AI model will **instantly classify** them as:

- `Wildfire` or `No Wildfire`

- If `Wildfire`, determine whether it's in a `green_area` or `desert`

---

### **HOW DOES IT WORK? 🛰️**
1️⃣ **Upload a satellite image** (JPG or PNG)

2️⃣ **Click 'Analyze Image'** to start the detection

3️⃣ **Get AI results:** `"Wildfire in Green Area"` or `"Nowildfire"`
""")

# Upload (we might add "choose random image from our library" here)
st.markdown("### 📸 Upload a Satellite Image for Analysis")
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "png", "jpeg"])

# Prediction
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔥 Analyze Image"):
        st.write("⏳ Processing your image... Please wait.")

        # Convering image
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes = img_bytes.getvalue()
        files = {"file": img_bytes}

        # STEP 1: Sending image --> Wildfire API
        wildfire_response = requests.post(WILDFIRE_API_URL, files=files)

        try:
            wildfire_response = requests.post(WILDFIRE_API_URL, files=files)
            wildfire_response.raise_for_status()
            wildfire_result = wildfire_response.json()
            wildfire_prediction = wildfire_result["wildfire_prediction"]
            wildfire_confidence = wildfire_result["wildfire_confidence"]


            st.success(f"🔥 Prediction: **{wildfire_prediction.upper()}**")
            st.info(f"✅ Confidence Score: **{wildfire_confidence * 100:.2f}%**")

            if wildfire_prediction == "wildfire":
                st.error("⚠️ ALERT! This image shows signs of a wildfire.")


                # STEP 2: Sending image to Environment API
                try:
                    environment_response = requests.post(ENVIRONMENT_API_URL, files=files)
                    environment_response.raise_for_status()
                    environment_result = environment_response.json()
                    environment_prediction = environment_result["environment_prediction"]

                    st.warning(f"🌍 Environment Type: **{environment_prediction}**")
                    st.success(f"🔎 Final Analysis: **Wildfire in {environment_prediction}**")
                except requests.exceptions.RequestException as e:
                    st.error(f"⚠️ Error fetching environment classification: {str(e)}")
            else:
                st.success("✅ No wildfire detected. Stay safe!")

        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ Error connecting to wildfire prediction API: {str(e)}")
