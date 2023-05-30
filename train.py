import cv2
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Set the directory path where the dataset is stored
dataset_path = "custom_dataset"

# Initialize variables
data = []
labels = []
target_names = []

# Load the dataset
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            image_path = os.path.join(root, file)
            label = os.path.basename(root)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (64, 64))  # Resize image to a consistent size
            data.append(image)
            labels.append(label)
            if label not in target_names:
                target_names.append(label)

# Convert data and labels to numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Perform one-hot encoding on the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels, num_classes=len(target_names))

# Split the data into training and testing sets
(train_data, test_data, train_labels, test_labels) = train_test_split(data, labels, test_size=0.2, random_state=42)

# Construct the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(target_names), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, batch_size=32, epochs=10, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)