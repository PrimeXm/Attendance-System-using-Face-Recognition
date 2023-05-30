import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import datetime

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label

# Initialize Firebase
cred = credentials.Certificate("Key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://realtime-b5552-default-rtdb.firebaseio.com/students'
})

# Function to retrieve attendance from Firebase and return the attendance details as a string
def retrieve_attendance():
    today = datetime.date.today().isoformat()
    attendance_ref = db.reference('attendance').child(today)
    attendance_data = attendance_ref.get()

    attendance_details = ""

    if attendance_data:
        attendance_details += f"Attendance for {today}:\n"
        for student_name, student_data in attendance_data.items():
            timestamp = student_data['timestamp']
            attendance_details += f"{student_name} - {timestamp}\n"
    else:
        attendance_details += f"No attendance data available for {today}."

    return attendance_details

# Main GUI class
class AttendanceApp(App):
    def build(self):
        # Create a vertical box layout
        layout = BoxLayout(orientation='vertical')

        # Create a label to display the attendance details
        attendance_label = Label(text=retrieve_attendance(), size_hint=(1, 1))

        # Add the label to the layout
        layout.add_widget(attendance_label)

        return layout

# Run the app
if __name__ == '__main__':
    AttendanceApp().run()
