import cv2
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import datetime

# Set the directory path where the dataset is stored
dataset_path = "custom_dataset"

# Initialize Firebase
cred = credentials.Certificate("Key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://realtime-b5552-default-rtdb.firebaseio.com/students'
})

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

# Function to update attendance in Firebase
def update_attendance(student_name, enrollment_number):
    today = datetime.date.today().isoformat()
    attendance_ref = db.reference('attendance').child(today)
    student_ref = attendance_ref.child(student_name)

    if not student_ref.get():
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        attendance_data = {
            'timestamp': timestamp,
            'enrollment_number': enrollment_number
        }
        student_ref.set(attendance_data)


# Function to get the enrollment number for a predicted label
def get_enrollment_number(predicted_label):
    student_ref = db.reference('students').child(predicted_label)
    student_data = student_ref.get()
    if student_data:
        enrollment_number = student_data.get('enrollment_no')
        return enrollment_number
    return None


# Initialize the video capture
video_capture = cv2.VideoCapture(0)

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize a frame counter
frame_counter = 0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Increment the frame counter
    frame_counter += 1

    # Skip face detection for some frames
    if frame_counter % 5 != 0:
        # Display the frame without face detection
        cv2.imshow('Video', cv2.resize(frame, (800, 600)))
        continue

    # Reset the frame counter after performing face detection
    frame_counter = 0

    # Convert the image from BGR color (OpenCV default) to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract the face region of interest
        face_image = frame[y:y + h, x:x + w]
        face_image = cv2.resize(face_image, (64, 64))
        face_image = face_image.astype("float32") / 255.0
        face_image = np.expand_dims(face_image, axis=0)

        # Perform face recognition using the trained model
        predicted_labels = model.predict(face_image)
        predicted_label_index = np.argmax(predicted_labels)
        predicted_distance = np.max(predicted_labels)

        # Check if the predicted distance is below the threshold
        if predicted_distance >= 0.5:
            predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
            enrollment_number = get_enrollment_number(predicted_label)
            if enrollment_number:
                # Label the face with name and enrollment number
                text = f"{predicted_label} - {enrollment_number}"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Update attendance for the recognized student
                update_attendance(predicted_label, enrollment_number)
                # Mark the student as attended
                cv2.putText(frame, "Marked", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            # Label the face as unknown
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', cv2.resize(frame, (800, 600)))

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()

# Destroy all windows
cv2.destroyAllWindows()
