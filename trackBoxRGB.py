import cv2 as cv
import numpy as np

# Start video capture
cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()

    # Define blue color range in RGB
    lower_blue = np.array([100, 0, 0])   # Adjust these values
    upper_blue = np.array([255, 50, 50]) # Adjust these values

    # Threshold the image to get only blue colors
    mask = cv.inRange(frame, lower_blue, upper_blue)

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
