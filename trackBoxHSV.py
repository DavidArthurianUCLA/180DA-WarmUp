import cv2 as cv
import numpy as np

# Start video capture
cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()

    # Convert from BGR to HSV color-space
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Blue color range in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    # Find contours
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Draw bounding box around the largest contour
    if contours:
        c = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv.imshow('frame', frame)

    # Break the loop when 'Esc' key is pressed
    if cv.waitKey(5) & 0xFF == 27:
        break

# Release the capture and close all windows
cap.release()
cv.destroyAllWindows()