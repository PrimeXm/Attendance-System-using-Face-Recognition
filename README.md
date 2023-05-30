# Attendance System using Face Recognition

The Attendance System using Face Recognition is a project that automates the process of marking attendance by using face recognition technology. It captures images from a camera, detects faces in the images, and matches them against a dataset of known faces to mark attendance.

## Features

- **Face detection**: The system uses computer vision techniques to detect faces in real-time captured frames.
- **Face recognition**: A machine learning model is trained on a dataset of known faces to recognize and match faces.
- **Database integration**: The system integrates with a database to store student information and attendance records.
- **Graphical User Interface (GUI)**: A user-friendly GUI allows for easy interaction and management of the attendance system.

## Prerequisites

Before running the project, make sure you have the following prerequisites:

- **Python**: Ensure that Python is installed on your system. You can download it from the official Python website: [https://www.python.org/](https://www.python.org/)
- **OpenCV**: Install the OpenCV library for Python. You can use the following command to install it using pip:
- pip install opencv-python
- - **Face Recognition Libraries**: Install the required face recognition libraries. For example, you can use the following command to install the face_recognition library:
- pip install face_recognition
- Firebase Admin SDK: Install the Firebase Admin SDK using the following command:
- pip install firebase-admin
- - Kivy: Install the Kivy framework using the following command:
- pip install kivy

I used pycharm 
also create own database to use it
## Usage

1. Clone the repository:

2. Install the required dependencies mentioned in the `Prerequisites` section.

3. Add your Firebase service account key file (`Key.json`) to the project directory.

4. Set up the Firebase Realtime Database and configure the `databaseURL` in the code to point to your Firebase database.

5. Run the application: main.py

6. Enter the student details in the GUI fields and click on the 'Start Capturing' button to begin capturing images for registration.

7. Follow the instructions displayed on the screen to capture images of the student's face from different angles.

8. Once the images are captured, they will be stored in the dataset directory and uploaded to Firebase along with the student's information.

9. The attendance system will use the trained CNN model to detect and recognize faces in real-time video frames. When a known face is detected, the attendance will be marked and updated in the Firebase database.

## Acknowledgments

- The face recognition model implementation in this project is inspired by the work of Adrian Rosebrock at PyImageSearch. You can find more details and tutorials on his website: [https://www.pyimagesearch.com/](https://www.pyimagesearch.com/)

## License

This project is licensed under the [MIT License](LICENSE).

