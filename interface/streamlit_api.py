import streamlit as st
import requests
from PIL import Image
import io

# FastAPI Endpoints
WILDFIRE_API_URL = "http://127.0.0.1:8001/predict-wildfire"  # Wildfire API
ENVIRONMENT_API_URL = "http://127.0.0.1:8002/predict-environment"  # Environment API

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

This service allows you to **upload satellite images**, and our AI model will **instantly classify** them as

- `"Wildfire"` or `"No Wildfire"`,

and determine if it's in a

- `"green_area"`, `"desert"`, `"water"`, or `"cloudy"`

region.

---

### **HOW DOES IT WORK? 🛰️**
1️⃣ **Upload a satellite image** (JPG or PNG)

2️⃣ **Click 'Analyze Image'** to start the detection

3️⃣ **Get AI results:** `"Wildfire in Green Area"` or `"Nowildfire in Desert"`
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

        if wildfire_response.status_code == 200:
            wildfire_result = wildfire_response.json()
            wildfire_prediction = wildfire_result["wildfire_prediction"]
            wildfire_confidence = wildfire_result["wildfire_confidence"]

            # STEP 2: Extracting enviro pred from wildfire API response
            environment_prediction = wildfire_result["environment_prediction"]
            final_result = wildfire_result["final_result"]

            # Displaying results
            st.success(f"🔥 Prediction: **{wildfire_prediction.upper()}**")
            st.info(f"✅ Confidence Score: **{wildfire_confidence * 100:.2f}%**")
            st.warning(f"🌍 Environment Type: **{environment_prediction}**")
            st.success(f"🔎 Final Analysis: **{final_result}**")

            # Displaying wildfire alert
            if wildfire_prediction == "wildfire":
                st.error("⚠️ ALERT! This image shows signs of a wildfire.")
            else:
                st.success("✅ No wildfire detected. Stay safe!")

        else:
            st.error(f"⚠️ Error: {wildfire_response.json().get('detail', 'Unknown error')}")
