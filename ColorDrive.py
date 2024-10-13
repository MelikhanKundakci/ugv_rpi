import cv2
import imutils
from picamera2 import Picamera2  # Library for accessing Raspberry Pi Camera
import numpy as np  # Library for mathematical calculations
import ipywidgets as widgets  # Library for creating interactive widgets such as buttons
import threading  # Library for creating new threads to execute tasks asynchronously
import time  # Import time for delays
from base_ctrl import BaseController  # Import the BaseController for robot control
from IPython.display import display, Image  # Import display function for Jupyter widgets


base = BaseController('/dev/serial0', 115200)

# Create a "Stop" button that users can click to stop the program
stopButton = widgets.ToggleButton(
    value=False,
    description='Stop',
    disabled=False,
    button_style='danger',  # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click to stop the program',
    icon='stop'  # (FontAwesome names without the `fa-` prefix)
)

# Define the display function to process video frames and recognize objects of specific colors
def view(button):
    camera = cv2.VideoCapture(-1)  # Use 0 for the default camera
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    color_upper = np.array([120, 255, 220])
    color_lower = np.array([90, 120, 90])
    min_radius = 12  # Define the minimum radius for detecting objects

    robot_moving = False  # Track whether the robot is currently moving

    display_handle = display(None, display_id=True)  # Create a display handle to update displayed images

    while True:
        ret, img = camera.read()
        if not ret:
            print("Failed to grab frame")
            break

        blurred = cv2.GaussianBlur(img, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, color_lower, color_upper)
        mask = cv2.erode(mask, None, iterations=5)
        mask = cv2.dilate(mask, None, iterations=5)

        # Find contours in the mask
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > min_radius:
                cv2.circle(img, (int(x), int(y)), int(radius), (128, 255, 255), 1)
                if not robot_moving:
                    # If the circle is detected and the robot is not moving, start moving forward
                    base.send_command({"T": 1, "L": 0.2, "R": 0.2})
                    robot_moving = True  # Set the flag to indicate the robot is moving
        else:
            if robot_moving:
                # If no circle is detected and the robot is moving, stop the robot
                base.send_command({"T": 1, "L": 0, "R": 0})
                robot_moving = False  # Set the flag to indicate the robot has stopped

        _, frame = cv2.imencode('.jpeg', img)
        display_handle.update(Image(data=frame.tobytes()))

        if stopButton.value:  # Check if the "Stop" button has been pressed
            base.send_command({"T": 1, "L": 0, "R": 0})  # Ensure the robot stops
            camera.release()  # Close the camera properly
            display_handle.update(None)  # Clear the displayed content
            print("Program stopped by user.")  # Inform the user that the program has been stopped
            break

# Display the "Stop" button and start a thread to execute the display function
display(stopButton)
thread = threading.Thread(target=view, args=(stopButton,))
thread.start()