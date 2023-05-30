import kivy
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from subprocess import Popen

# Create the main app class
class MyApp(App):

    # Function to handle button click events
    def button_click(self, button_num):
        if button_num == 1:
            # Run Script 1
            Popen(["python", "Add.py"])
        elif button_num == 2:
            # Run Script 2
            Popen(["python", "attendance2.0.py"])
        elif button_num == 3:
            # Run Script 3
            Popen(["python", "getinfo.py"])

    # Build the app UI
    def build(self):
        # Create a float layout
        layout = FloatLayout()

        # Create the buttons
        button1 = Button(text='Register new student', size_hint=(None, None), size=(300, 100), pos_hint={'center_x': 0.5, 'center_y': 0.6})
        button1.bind(on_press=lambda x: self.button_click(1))

        button2 = Button(text='Update Attendance', size_hint=(None, None), size=(300, 100), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        button2.bind(on_press=lambda x: self.button_click(2))

        button3 = Button(text='Get Attendance', size_hint=(None, None), size=(300, 100), pos_hint={'center_x': 0.5, 'center_y': 0.4})
        button3.bind(on_press=lambda x: self.button_click(3))

        # Add the buttons to the layout
        layout.add_widget(button1)
        layout.add_widget(button2)
        layout.add_widget(button3)

        return layout

# Run the app
if __name__ == '__main__':
    MyApp().run()
