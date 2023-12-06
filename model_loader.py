import tensorflow as tf

PLANTS = ["Apple", "Bell Pepper", "Potato", "Tomato", "Peach"]

DISEASES = {
    "Apple": ["Apple Scab", "Black Rot", "Cedar Black Rust", "Healthy"],
    "Bell Pepper": ["Bacterial Spot", "Healthy"],
    "Peach": ["Bacterial Spot", "Healthy"],
    "Potato": ["Early Blight", "Healthy", "Late Blight"],
    "Tomato": ["Bacterial Spot", "Early Blight", "Healthy", "Late Blight", "Septoria Leaf Spot", "Yellow Leaf Curl Virus"]
}

dir_path = "Models"


def load_model(plant):
    model = tf.keras.models.load_model(filepath=f"{dir_path}/{plant}")
    return model


def get_plants():
    return PLANTS


def get_disease(plant):
    return DISEASES.get(plant)
