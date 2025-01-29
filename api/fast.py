import io
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

from enviro_class.modeling import load_model

app = FastAPI()

# optional, but I added it since it is recommended
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Loading model
app.state.model = load_model()

# Image preprocessing
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image to match Keras model input format."""
    image = image.resize((224, 224))  # Ensuring correct input size
    image = img_to_array(image) / 255.0  # Normalizing pixel values
    image = np.expand_dims(image, axis=0)  # Adding batch dimension
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict if uploaded satellite image shows signs of wildfire
    """

    if app.state.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Reading + preprocessing uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Convert to RGB
        input_tensor = preprocess_image(image)

        # Predicting
        predictions = app.state.model.predict(input_tensor)
        class_index = int(np.argmax(predictions, axis=-1)[0])
        confidence = float(np.max(predictions))

        # Class labels
        class_labels = ["nowildfire", "wildfire"]
        prediction = class_labels[class_index]

        return {"prediction": prediction, "confidence": round(confidence, 4)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/")
def index():
    return {"message": "EnviroClass API is running!"}
