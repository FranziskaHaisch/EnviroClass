import io
import os
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

from enviro_class.modeling import load_wildfire_model, load_environment_model

app = FastAPI()

def get_port():
    return int(os.getenv("PORT", 8000))

# optional, but I added it since it is recommended
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Loading WILDFIRE DETECTION + ENVIRONMENT model
app.state.wildfire_model = load_wildfire_model()
app.state.environment_model = load_environment_model()

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

        # STEP 1: Predicting WILDFIRE
        predictions = app.state.wildfire_model.predict(input_tensor)
        print("Raw model output:", predictions)  # Debugging

        class_index = int(np.argmax(predictions, axis=-1)[0]) # based on softmax
        print("Predicted class index:", class_index)  # Debugging

        # STEP 2: Labelling + results
        wildfire_labels = ["nowildfire", "wildfire"]
        wildfire_prediction = wildfire_labels[class_index]
        wildfire_confidence = float(predictions[0][0])

        return {
            "wildfire_prediction": wildfire_prediction,
            "wildfire_confidence": wildfire_confidence
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

# Preprocessing function for environment classification
def preprocess_environment_image(image: Image.Image) -> np.ndarray:
    image = image.resize((64, 64))  # Resize to model's expected input size
    image = img_to_array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)
    return image


@app.post("/predict-environment")
async def predict_environment(file: UploadFile = File(...)):
    """Predicting if satellite image contains e.g. 'green_area', 'desert'"""

    if app.state.environment_model is None:
        raise HTTPException(status_code=500, detail="Environment model not loaded")

    #try:
    # STEP 0: Reading + preprocessing uploaded image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess_environment_image(image)

    # STEP 1: Predicting ENVIROMENT
    predictions = app.state.environment_model(input_tensor)
    class_index = int(np.argmax(predictions, axis=-1)[0]) # only works if each bounding box contains individual class score

    # STEP 2: Labelling + results
    environment_labels = ['Agriculture',
                        'Airport',
                        'Beach',
                        'City',
                        'Desert',
                        'Forest',
                        'Grassland',
                        'Highway',
                        'Lake',
                        'Mountain',
                        'Parking',
                        'Port',
                        'Railway',
                        'River']# MUST BE ADAPTED TO OUR MODEL
    environment_prediction = environment_labels[class_index]

    return {"prediction": environment_prediction}

    #except Exception as e:
        #raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/")
def index():
    return {"message": "Wildfire & Environment Analysis API is running!"}
