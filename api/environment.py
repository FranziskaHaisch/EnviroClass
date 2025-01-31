import io
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

from enviro_class.modeling import load_environment_model

app = FastAPI()

# optional, but I added it since it is recommended
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loading ENVIRONMENT model
app.state.environment_model = load_environment_model()

# Image preprocessing
def preprocess_environment_image(image: Image.Image) -> np.ndarray:
    """Preprocessing image for ENVIRONMENT model using current shape of:(64,64,3)."""
    image = image.resize((64,64))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict-environment")
async def predict_environment(file: UploadFile = File(...)):
    """Predicting if satellite image contains e.g. 'green_area', 'desert'"""

    if app.state.environment_model is None:
        raise HTTPException(status_code=500, detail="Environment model not loaded")

    try:
        # Reading image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = preprocess_environment_image(image)

        # Predicting
        predictions = app.state.environment_model.predict(input_tensor)
        class_index = int(np.argmax(predictions, axis=-1)[0]) # SHOULD BE SIGMOID???

        # Class labels
        environment_labels = ["cloudy", "green_area", "water", "desert"] # MUST BE ADAPTED TO OUR MODEL
        environment_prediction = environment_labels[class_index]

        return {"prediction": environment_prediction}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/")
def index():
    return {"message": "Environment Analysis API is running!"}
