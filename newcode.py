import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

# Lower and upper bounds of each color in HSV space
lower = {'red': ([166, 84, 141]), 'green': ([50, 50, 120]), 'blue': ([97, 100, 117])}
upper = {'red': ([186, 255, 255]), 'green': ([70, 255, 255]), 'blue': ([117, 255, 255])}

# BGR tuple of each color
colors = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0)}

# Flag to close the window after 5 seconds
close_window_flag = False

# Function to process the frame
def process_frame(frame):
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mlist = []
    clist = []
    ks = []

    for (key, value) in upper.items():
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.inRange(hsv, np.array(lower[key]), np.array(upper[key]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mlist.append(mask)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(cnts) >= 1:
            clist.append(cnts[-1])
            ks.append(key)

    for i, cnt in enumerate(clist):
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(frame, [approx], 0, (0), 2)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        

        if len(approx) == 4:
            x2 = approx.ravel()[2]
            y2 = approx.ravel()[3]
            x4 = approx.ravel()[6]
            y4 = approx.ravel()[7]
            side1 = abs(x2 - x)
            side2 = abs(y4 - y)

            if abs(side1 - side2) <= 2:
                if ks[i] == 'red':
                    weight_value = 'L'
                    close_window_flag = True
                    cv2.putText(frame, f"{ks[i]} Square: {weight_value}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[ks[i]], 2)
                    break

                elif ks[i] == 'blue':
                    weight_value = 'M'
                    close_window_flag = True
                    cv2.putText(frame, f"{ks[i]} Square: {weight_value}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[ks[i]], 2)
                    break

                elif ks[i] == 'green':
                    weight_value = 'S'
                    cv2.putText(frame, f"{ks[i]} Square: {weight_value}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[ks[i]], 2)
                    close_window_flag = True
                    break

                else:
                    weight_value = "unknown"  # Add a default value for other colors
                    cv2.putText(frame, f"Square: {weight_value}kg", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[ks[i]], 2)
            

    return frame

# Streamlit app
st.title('Shape Detection with OpenCV and Streamlit')

# Open webcam
cap = cv2.VideoCapture(0)

# Create an empty placeholder for the image
webcam_placeholder = st.empty()

# Streamlit loop
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Cannot receive frame from webcam.")
        break

    # Process the frame
    processed_frame = process_frame(frame)

    # Display the frame in Streamlit
    webcam_placeholder.image(processed_frame, channels='BGR')

    # Check if the close_window_flag is set to True
    if close_window_flag:
        break

    # Add a small delay to make it real-time
    time.sleep(0.1)

# Close the window after 5 seconds
time.sleep(5)
cap.release()

