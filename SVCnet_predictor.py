
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Parameters
IMG_SIZE = 224

# Load the saved models
resnet = load_model("resnet_feature_extractor.h5")
clf = joblib.load("fall_detection_model.pkl")

# Feature extraction function
def extract_features(images):
    images = preprocess_input(images)
    return resnet.predict(images)

# Prediction function for a single image
def predict_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        return "Invalid Image Path"

    # Resize and preprocess the image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, axis=0)

    # Extract features
    img_features = extract_features(img)

    # Predict using the classifier
    prediction = clf.predict(img_features)
    return "Fall" if prediction[0] == 1 else "Not Fall"

# Example Usage
image_path = 'fall/4_cropped_0.jpg'  # Replace with the path to your test image
result = predict_image(image_path)
print(f"The prediction for the image is: {result}")