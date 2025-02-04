import tensorflow as tf
import os
from ultralytics import YOLO


def load_wildfire_model():
    """Loading WILDFIRE model from CORRECT location"""

    # absolute path
    absolute_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Moves up one level

    # path to wildfire model
    wildfire_model_path = os.path.join(absolute_path, "models", "wildfire_cnn_2classes.keras")

    # Loading +returning model
    return tf.keras.models.load_model(wildfire_model_path)


# Loading ENVIRONMENT CLASSIFICATION model
def load_environment_model():
    """Loading ENVIRONMENT model from CORRECT location"""
    # absolute path
    absolute_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Moves up one level

    # path to environment model
    environment_model_path = os.path.join(absolute_path, "models", "environment_yolo_14classes.pt")

    # Loading +returning model
    return YOLO(environment_model_path)
