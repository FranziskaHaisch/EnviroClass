import io
import os
import base64
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import torch
from ultralytics import YOLO

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
app.state.wildfire_model = load_wildfire_model() # CNN
app.state.environment_model = load_environment_model() # YOLO


# PART 1: WILDFIRE DETECTION - CNN MODEL
# Image preprocessing
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocessing image to match WILDFIRE model input format
    WE MIGHT HAVE TO ADAPT THIS DEPENDING ON OUR MODEL"""
    image = image.resize((350,350))  # Ensuring correct input size!!!!
    image = np.array(image, dtype=np.float32)  # Converting to NumPy array
    image = np.expand_dims(image, axis=0)  # Adding batch dimension (1, 350, 350, 3)
    return image


@app.post("/predict-wildfire")
async def predict_wildfire(file: UploadFile = File(...)):
    """
    Predicting if uploaded satellite image shows signs of wildfire using CNN
    """

    if app.state.wildfire_model is None:
        raise HTTPException(status_code=500, detail="Wildfire model not loaded")

    try:
        # STEP 0: Reading + preprocessing uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Convert to RGB
        input_tensor = preprocess_image(image)

        # STEP 1: Predicting WILDFIRE
        wf_predictions = app.state.wildfire_model.predict(input_tensor)
        print("Raw model output:", wf_predictions)  # Debugging

        # STEP 2: Labelling + results
        wildfire_labels = ["wildfire", "nowildfire"]
        wildfire_probability = float(wf_predictions[0][0]) # prob score
        wildfire_prediction = "wildfire" if wildfire_probability > 0.5 else "nowildfire"

        return {
            "wildfire_prediction": wildfire_prediction,
            "wildfire_confidence": wildfire_probability if wildfire_prediction == "wildfire" else (1 - wildfire_probability)
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


# PART 2: ENVIRONMENT DETECTION - YOLO MODEL
# Preprocessing function for environment classification
def preprocess_environment_image(image: Image.Image) -> np.ndarray:
    image = image.resize((640, 640))  # Resize to model's expected input size
    return image


@app.post("/predict-environment")
async def predict_environment(file: UploadFile = File(...)):
    """
    Predicting if satellite image contains specific areas using YOLO
    """

    if app.state.environment_model is None:
        raise HTTPException(status_code=500, detail="Environment model not loaded")

    try:
        # STEP 0: Reading + preprocessing uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = preprocess_environment_image(image)

            # STEP 1: Predicting ENVIROMENT with YOLO
        env_prediction = app.state.environment_model.predict(input_tensor)

        # STEP 2: Extracting first detected class (if any)
        if len(env_prediction[0].boxes) > 0:
            detected_objects = []
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
                                  'River']

            for obj in env_prediction[0].boxes:
                class_index = int(obj.cls.item())
                confidence = float(obj.conf.item())
                detected_class = environment_labels[class_index]

                # Modify "Lake" output to "Lake or Sea"
                if detected_class == "Lake":
                    detected_class = "Lake or Sea"

                detected_objects.append({
                    "class": detected_class,
                    "confidence": confidence
                })

        # STEP 5: Generating + saving annotated image + bounding boxes
            output_image = env_prediction[0].plot()
            output_image_path = "annotated_image.jpg"
            output_image_pil = Image.fromarray(output_image)
            output_image_pil.save(output_image_path)

        # STEP 5: Converting image --> base64 for API response
            with open(output_image_path, "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode("utf-8")

            return {
                "environment_prediction": detected_objects,
                "confidence": confidence,
                "annotated_image": encoded_image
            }

        else:
            return {"message": "No significant environment features detected."}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")



@app.get("/")
def index():
    return {"message": "Wildfire & Environment Analysis API is running!"}
