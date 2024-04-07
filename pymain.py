import socketio
import os
import cv2
import numpy as np
import pickle
import os
import json
from shutil import copyfile
import tensorflow as tf
import firebase_admin
from firebase_admin import credentials, storage

cred = credentials.Certificate("botanicrx-3bf34-firebase-adminsdk-2mxfj-ad3a9e0d5b.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'botanicrx-3bf34.appspot.com'
})

bucket = storage.bucket()


local_folder_path = "Predicted_output"  # Path to the local folder to be uploaded
remote_base_path = "Plants/"  # Remote folder path in Firebase Storage


# Load the trained model from the TensorFlow SavedModel
model = tf.saved_model.load('saved_model')

# Load class names from the JSON file
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

image_dir = "C:\\Users\\aadhi\\Documents\\Projects\\Module1\\received_images"
output_dir = "C:\\Users\\aadhi\\Documents\\Projects\\Module1\\Predicted_output"

sio = socketio.Client()

d = {}

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
    upload_folder(local_folder_path, remote_base_path)


def upload_folder(local_folder_path, remote_base_path):
    for root, _, files in os.walk(local_folder_path):
        for filename in files:
            local_file_path = os.path.join(root, filename)
            remote_file_path = os.path.relpath(local_file_path, local_folder_path).replace("\\", "/")
            remote_file_path = os.path.join(remote_base_path, remote_file_path)

            # Create the parent directories if they don't exist
            remote_dir = os.path.dirname(remote_file_path)
            if remote_dir:
                blob = bucket.blob(remote_dir + "/")  # Create a Blob object for the directory
                blob.upload_from_string("")  # Upload an empty string to create the directory

            # Upload the file
            blob = bucket.blob(remote_file_path)
            blob.upload_from_filename(local_file_path)


@sio.event
def disconnect():
    print('disconnected from server')

sio.connect('http://localhost:5000')

sio.wait()