import cv2
import numpy as np
import pickle
import json

# Load the trained model from the pickle file
with open('leaf_classification_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load class names from the JSON file
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

image_dir = "C:\\Users\\aadhi\\Documents\\Projects\\Module1\\Training"

for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        image_path = os.path.join(image_dir, filename)
        new_image = cv2.imread(image_path)
        new_image = cv2.resize(new_image, (224, 224))
        new_image = new_image / 255.0  # Normalize pixel values to [0, 1]
        new_image = np.expand_dims(new_image, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(new_image)
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = class_names[str(predicted_class_index)]
        print(f'Predicted class: {predicted_class_name}')

