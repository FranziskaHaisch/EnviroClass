import tensorflow as tf
import os


model = tf.keras.models.load_model("models/test_model.keras")
model.save("models/test_model.keras", include_optimizer=False)


def load_wildfire_model():
    """Loading WILDFIRE model from CORRECT location"""

    # absolute path
    absolute_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Moves up one level

    # path to wildfire model
    wildfire_model_path = os.path.join(absolute_path, "models", "cnn_2classes_ver1.keras")

    # Loading +returning model
    return tf.keras.models.load_model(wildfire_model_path)


# Loading ENVIRONMENT CLASSIFICATION model
def load_environment_model():
    """Loading ENVIRONMENT model from CORRECT location"""
    # absolute path
    absolute_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Moves up one level

    # path to environment model
    environment_model_path = os.path.join(absolute_path, "models", "test_model.keras")

    # Loading +returning model
    return tf.keras.models.load_model(environment_model_path)
