import io
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import requests
import os

from enviro_class.modeling import load_wildfire_model

app = FastAPI()

# optional, but I added it since it is recommended
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Loading WILDFIRE DETECTION model
app.state.wildfire_model = load_wildfire_model()

# Image preprocessing (we need to adapt!!!)
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocessing image to match WILDFIRE model input format
    WE MIGHT HAVE TO ADAPT THIS DEPENDING ON OUR MODEL"""
    image = image.resize((350,350))  # Ensuring correct input size!!!!
    image = img_to_array(image) / 255.0  # Normalizing pixel values
    image = np.expand_dims(image, axis=0)
    #image = image.reshape(1, -1)  # Flattening (1, 64*64*3)
    return image

@app.post("/predict-wildfire")
async def predict_wildfire(file: UploadFile = File(...)):
    """
    Predicting if uploaded satellite image shows signs of wildfire
    """

    if app.state.wildfire_model is None:
        raise HTTPException(status_code=500, detail="Wildfire model not loaded")

    try:
        # STEP 0: Reading + preprocessing uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Convert to RGB
        input_tensor = preprocess_image(image)

        # STEP 1: Predicting Wildfire
        predictions = app.state.wildfire_model.predict(input_tensor)
        print("Raw model output:", predictions)  # Debugging

        class_index = int(np.argmax(predictions, axis=-1)[0]) # SHOULD BE SIGMOID???
        print("Predicted class index:", class_index)  # Debugging

        # Class labels
        wildfire_labels = ["nowildfire", "wildfire"]
        wildfire_prediction = wildfire_labels[class_index]
        wildfire_confidence = float(np.max(predictions))

        # STEP 2: Forwarding image + wildfire prediction to 2nd API
        environment_api_url = "http://127.0.0.1:8002/predict-environment"
        files = {"file": ("image.jpg", io.BytesIO(image_bytes), "image/jpeg")}
        response = requests.post(environment_api_url, files=files)

        # Check if 2nd API responded successfully
        if response.status_code == 200:
            environment_result = response.json()
            environment_prediction = environment_result.get("prediction", "unknown")
        else:
            environment_prediction = "unknown"

        # STEP 3: Combining both predictions

        final_result = f"{wildfire_prediction} in {environment_prediction}"

        return {
            "wildfire_prediction": wildfire_prediction,
            "wildfire_confidence": round(wildfire_confidence, 4),
            "environment_prediction": environment_prediction,
            "final_result": final_result
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/")
def index():
    return {"message": "EnviroClass API is running!"}





# # Loading WILDFIRE DETECTION model
# app.state.wildfire_model = load_model()

# # Image preprocessing (we need to adapt!!!)
# def preprocess_image(image: Image.Image) -> np.ndarray:
#     """Preprocessing image to match WILDFIRE model input format
#     WE MIGHT HAVE TO ADAPT THIS DEPENDING ON OUR MODEL"""
#     image = image.resize((64, 64))  # Ensuring correct input size!!!!
#     image = img_to_array(image) / 255.0  # Normalizing pixel values
#     image = np.expand_dims(image, axis=0)
#     return image

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     """
#     Predict if uploaded satellite image shows signs of wildfire
#     """

#     if app.state.wildfire_model is None:
#         raise HTTPException(status_code=500, detail="Wildfire model not loaded")

#     try:
#         # Reading + preprocessing uploaded image (we need to adapt!)
#         image_bytes = await file.read()
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Convert to RGB
#         input_tensor = preprocess_image(image)

#         # Predicting
#         predictions = app.state.model.predict(input_tensor)
#         print("Raw model output:", predictions)  # Debugging

#         class_index = int(np.argmax(predictions, axis=-1)[0])
#         print("Predicted class index:", class_index)  # Debugging

#         # Class labels (VALID RANGE???!)
#         class_labels = ["cloudy", "green_area", "water", "desert"]

#         if class_index >= len(class_labels) or class_index < 0:
#             raise ValueError(f"Invalid class index {class_index}. Model output: {predictions}")

#         prediction = class_labels[class_index]
#         confidence = float(np.max(predictions))

#         return {"prediction": prediction, "confidence": round(confidence, 4)}

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


# @app.get("/")
# def index():
#     return {"message": "EnviroClass API is running!"}
