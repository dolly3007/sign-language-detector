import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("best_model.keras")

# Load and preprocess the image
img_path = "C_test.jpg"  # Replace with a real test image path
img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"Image not found at {img_path}")

# Resize and convert to RGB
img = cv2.resize(img, (128, 128))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Model expects shape (1, 128, 128, 3)
img_array = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)
confidence = np.max(prediction)

# Class labels (match the training class order)
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

print(f"Predicted Label: {class_names[predicted_class]} (Confidence: {confidence:.2f})")
