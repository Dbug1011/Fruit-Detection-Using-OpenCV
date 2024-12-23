import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model

# Ensure TensorFlow optimizations do not interfere
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Check TensorFlow version and current working directory
print(f"TensorFlow Version: {tf.__version__}")
print(f"Current Working Directory: {os.getcwd()}")

# Load your trained model
model = load_model("trained_model (3).h5")

# Define the labels for your model
class_labels = [
    "apple", "banana", "beetroot", "bell pepper", "cabbage", "capsicum", "carrot",
    "cauliflower", "chilli pepper", "corn", "cucumber", "eggplant", "garlic", "ginger",
    "grapes", "jalepeno", "kiwi", "lemon", "lettuce", "mango", "onion", "orange",
    "paprika", "pear", "peas", "pineapple", "pomegranate", "potato", "raddish",
    "soy beans", "spinach", "sweetcorn", "sweetpotato", "tomato", "turnip", "watermelon"
]

# Preprocessing function
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (64, 64))  # Resize to the model's input size
    normalized_frame = resized_frame / 255.0  # Normalize pixel values to [0, 1]
    return np.expand_dims(normalized_frame, axis=0)  # Add batch dimension

# Start webcam capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access webcam.")
    exit()

print("Press 'q' to quit.")

# Process frames from the webcam
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from webcam.")
        break

    # Preprocess the frame
    input_frame = preprocess_frame(frame)

    # Make a prediction
    predictions = model.predict(input_frame)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    # Display the prediction on the frame
    text = f"{predicted_class}: {confidence:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Real-Time Object Detection', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()