import socketio
import os
import cv2
import numpy as np
import pickle
import os
import json
from shutil import copyfile
import tensorflow as tf

# Load the trained model from the TensorFlow SavedModel
model = tf.saved_model.load('saved_model')

# Load class names from the JSON file
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

image_dir = "C:\\Users\\aadhi\\Documents\\Projects\\Module1\\received_images"
output_dir = "C:\\Users\\aadhi\\Documents\\Projects\\Module1\\Predicted_output"

sio = socketio.Client()

completed_folder = 'received_images'  # Specify the folder where completed images will be stored

if not os.path.exists(completed_folder):
    os.makedirs(completed_folder)

@sio.event
def connect():
    print('connection established')

@sio.on('image')
def receive_image(data):
    filename = data['filename']
    image_data = data['data']
    image_path = os.path.join(completed_folder, filename)
    
    # Write the received image data to a file in the completed folder
    with open(image_path, 'wb') as file:
        file.write(image_data)
    
    # Process the received image
    new_image = cv2.imread(image_path)
    new_image = cv2.resize(new_image, (224, 224))
    new_image = new_image.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
    new_image = np.expand_dims(new_image, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.signatures['serving_default'](tf.constant(new_image))['dense_1']
    predicted_class_index = np.argmax(predictions.numpy())
    predicted_class_name = class_names[str(predicted_class_index)]
    print(f'Predicted class: {predicted_class_name}')

    # Create folder if it doesn't exist
    class_dir = os.path.join(output_dir, predicted_class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # Move the image to the predicted class folder
    output_path = os.path.join(class_dir, filename)
    copyfile(image_path, output_path)

@sio.event
def disconnect():
    print('disconnected from server')

sio.connect('http://localhost:5000')

sio.wait()