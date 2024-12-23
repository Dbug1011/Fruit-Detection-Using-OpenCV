import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import cv2
import numpy as np
import time

# Check GPU availability and enable memory growth
def check_gpu():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.set_memory_growth(gpu, True)
            print("GPU memory growth set.")
        except RuntimeError as e:
            print(e)

check_gpu()

np.set_printoptions(suppress=True)

# Load the model
try:
    model = tf.keras.models.load_model("trained_model (3).h5")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load test dataset and class names
try:
    test_set = tf.keras.utils.image_dataset_from_directory(
        "test",
        labels='inferred',
        label_mode='categorical',
        class_names=None,
        color_mode='rgb',
        batch_size=32,
        image_size=(64, 64),
        shuffle=True,
        interpolation='bilinear',
    )
    class_names = test_set.class_names
    print(f"Class names: {class_names}")
except Exception as e:
    print(f"Error loading test dataset: {e}")
    exit()

# Save class names to 'labels.txt'
with open("labels.txt", "w") as file:
    for name in class_names:
        file.write(f"{name}\n")

# Ensure input size matches training
size = (64, 64)  # Model input size

# Start webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

fps_values = []  # Store FPS values to calculate average FPS

# Function to process the frame and predict
def process_frame(frame):
    # Save the frame to a temporary file
    temp_image_path = "temp_frame.jpg"
    cv2.imwrite(temp_image_path, frame)

    # Load the saved frame as an image for prediction
    image = tf.keras.preprocessing.image.load_img(temp_image_path, target_size=size)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch

    # Perform prediction
    predictions = model.predict(input_arr)
    class_index = np.argmax(predictions[0])
    class_name = class_names[class_index]
    confidence_score = predictions[0][class_index]

    # Print predictions
    print(predictions)

    return class_name, confidence_score

while cap.isOpened():
    start = time.time()

    # Capture a frame from the webcam
    ret, img = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Predict the class and confidence score
    class_name, confidence_score = process_frame(img)

    # Calculate FPS
    end = time.time()
    total_time = end - start
    fps = 1 / total_time
    fps_values.append(fps)

    # Average FPS over the last 10 frames
    if len(fps_values) > 10:
        fps_values.pop(0)
    avg_fps = np.mean(fps_values)

    # Display FPS, class name, and confidence score on the frame
    cv2.putText(img, f'FPS: {int(avg_fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(img, f"Class: {class_name}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(img, f"Confidence: {confidence_score * 100:.2f}%", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Webcam Classification', img)

    # Break on pressing 'ESC'
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
