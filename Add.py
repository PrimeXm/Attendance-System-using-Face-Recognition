import cv2
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button

# Initialize Firebase app
cred = credentials.Certificate("Key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://realtime-b5552-default-rtdb.firebaseio.com/'
})

# Create a reference to the Firebase database
db_ref = db.reference()

# Set the directory path where you want to save the dataset
dataset_path = "custom_dataset"

# Create the face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the kv file for GUI layout
Builder.load_string('''
<MainWindow>:
    orientation: 'vertical'
    padding: 30
    spacing: 20

    Label:
        text: 'Student Name:'
        font_size: '20sp'
        size_hint_y: None
        height: self.texture_size[1] + dp(10)

    TextInput:
        id: name_input
        font_size: '18sp'
        size_hint_y: None
        height: dp(40)

    Label:
        text: 'Enrollment Number:'
        font_size: '20sp'
        size_hint_y: None
        height: self.texture_size[1] + dp(10)

    TextInput:
        id: enroll_input
        font_size: '18sp'
        size_hint_y: None
        height: dp(40)

    Label:
        text: 'Department:'
        font_size: '20sp'
        size_hint_y: None
        height: self.texture_size[1] + dp(10)

    TextInput:
        id: department_input
        font_size: '18sp'
        size_hint_y: None
        height: dp(40)

    Button:
        text: 'Start Capturing'
        font_size: '20sp'
        size_hint_y: None
        height: dp(40)
        background_normal: ''
        background_color: (0.2, 0.6, 0.9, 1)

        canvas.before:
            Color:
                rgba: self.background_color
            RoundedRectangle:
                pos: self.pos
                size: self.size
                radius: [dp(20),]

        on_press: root.start_capturing()
''')


class MainWindow(BoxLayout):
    def start_capturing(self):
        # Get the student details from the input fields
        student_name = self.ids.name_input.text
        enrollment_no = self.ids.enroll_input.text
        student_department = self.ids.department_input.text

        # Create a reference to the student's folder in Firebase
        student_ref = db_ref.child('students').child(student_name)

        # Set the student details in Firebase
        student_data = {
            'enrollment_no': enrollment_no,
            'department': student_department
        }
        student_ref.set(student_data)

        # Set the student directory path for saving images
        student_directory = os.path.join(dataset_path, student_name)

        # Create the student directory if it doesn't exist
        if not os.path.exists(student_directory):
            os.makedirs(student_directory)

        # Start the camera capture
        camera = cv2.VideoCapture(0)

        num_images_captured = 0

        while num_images_captured < 100:
            # Read the current frame from the camera
            ret, frame = camera.read()

            # Convert the frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Register New Student", frame)

            # Capture and save images when 's' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('s'):
                if len(faces) == 1:
                    # Increment the number of images captured for the current student
                    num_images_captured += 1

                    # Save the captured face image with a unique ID
                    image_id = f"{student_name}_{num_images_captured}"
                    image_path = os.path.join(student_directory, f"{image_id}.jpg")
                    cv2.imwrite(image_path, gray_frame[y:y + h, x:x + w])

                    print(f"Captured image {num_images_captured} for student {student_name}")

                    # Store image details in Firebase with a unique ID
                    image_data = {
                        'enrollment_no': enrollment_no,
                        'department': student_department,
                        'image_path': image_path
                    }
                    student_ref.child('images').push(image_data)

            # Quit the program when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera
        camera.release()

        # Close the OpenCV window
        cv2.destroyAllWindows()


class StudentRegistrationApp(App):
    def build(self):
        return MainWindow()


if __name__ == '__main__':
    StudentRegistrationApp().run()
