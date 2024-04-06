import os
import cv2
import numpy as np
import pickle
from tensorflow import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf


# Set image size
IMAGE_SIZE = (224, 224)

# Function to load and preprocess the dataset
def load_dataset(root_dir, subset):
    data = []
    labels = []
    class_names = {}
    for label, species_dir in enumerate(os.listdir(os.path.join(root_dir, subset))):
        class_names[label] = species_dir
        species_path = os.path.join(root_dir, subset, species_dir)
        for image_file in os.listdir(species_path):
            image_path = os.path.join(species_path, image_file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, IMAGE_SIZE)
            image = image / 255.0  # Normalize pixel values to [0, 1]
            data.append(image)
            labels.append(label)
    # Save class names to a file
    import json

    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)

    return np.array(data), np.array(labels), class_names

# Define the root directory of your dataset
root_dir = "C:\\Users\\aadhi\\Documents\\Projects\\Module1\\Segmented Medicinal Leaf Images"

# Load the training dataset
train_data, train_labels, class_names = load_dataset(root_dir, 'train')

# Split the training dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2)

# Define and build your deep learning model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(np.max(train_labels) + 1, activation='softmax')  # Use np.max(train_labels) + 1 as the number of output classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# model.sumamry()

# Save the trained model as a pickle file
# with open('leaf_classification_model.pkl', 'wb') as f:
#     pickle.dump(model, f)
# Save the model
tf.saved_model.save(model, 'saved_model')

# Load the testing dataset
# test_data, test_labels, class_names = load_dataset(root_dir, 'test')

# # Evaluate the model on the testing dataset
# test_loss, test_acc = model.evaluate(test_data, test_labels)
#print(f'Test accuracy: {test_acc * 100:.2f}%')

# image_dir = "C:\\Users\\aadhi\\Documents\\Projects\\Module1\\Training"




# image_dir = "C:\\Users\\aadhi\\Documents\\Projects\\Module1\\prep"
# for filename in os.listdir(image_dir):
#     if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
#         # Predict on new images
#         new_image_path = os.path.join(image_dir, filename)
#         new_image = cv2.imread(new_image_path)
#         new_image = cv2.resize(new_image, IMAGE_SIZE)
#         new_image = new_image / 255.0  # Normalize pixel values to [0, 1]
#         new_image = np.expand_dims(new_image, axis=0)  # Add batch dimension

#         # Make predictions
#         predictions = model.predict(new_image)
#         predicted_class_index = np.argmax(predictions)
#         predicted_class_name = class_names[predicted_class_index]
#         print(f'Predicted class: {predicted_class_name}')
