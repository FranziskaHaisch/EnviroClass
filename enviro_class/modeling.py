import tensorflow as tf

def load_model():
    """Loading our pre-trained model."""
    model = tf.keras.models.load_model("enviro_class/model")  # We need to update the path !!!!
    return model
