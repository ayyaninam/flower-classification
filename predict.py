import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model("iris_classification_model.h5")

def preprocess_image_for_prediction(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    # Resize the image to the size used in training (150x150)
    image = cv2.resize(image, (150, 150))
    # Normalize the image
    image = image / 255.0
    # Expand dimensions to match the input shape of the model
    image = np.expand_dims(image, axis=0)
    return image

def classify_iris_species(image_path):
    # Preprocess the input image
    image = preprocess_image_for_prediction(image_path)
    # Predict the class
    predictions = model.predict(image)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]

    # Define the class labels
    class_labels = ["iris-setosa", "iris-versicolour", "iris-virginica"]
    species = class_labels[class_index]

    return species, confidence

# Example usage
image_path = "/Users/apple/Downloads/Iris_virginica_2 (1).jpg"  # Provide the path to an image
species, confidence = classify_iris_species(image_path)
print(f"Predicted species: {species} with confidence: {confidence*100}%")
