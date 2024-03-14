import cv2
import numpy as np

# Load the image
image = cv2.imread('C:/Users/lingz/Downloads/linePhoto.jpg')  # Make sure to provide the correct path
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assuming the largest contour is your "L" shape
L_contour = max(contours, key=cv2.contourArea)

# You might need a custom function to analyze the contour and find the corner of the "L"
# For demonstration, let's say you've identified the corner and two points along the legs
corner_point = (x1, y1)
leg1_point = (x2, y2)
leg2_point = (x3, y3)

# Draw the legs of the "L"
cv2.line(image, corner_point, leg1_point, (255, 0, 0), thickness=2)
cv2.line(image, corner_point, leg2_point, (255, 0, 0), thickness=2)

# Highlight the corner point, which is where the 90-degree angle is
cv2.circle(image, corner_point, radius=5, color=(0, 255, 0), thickness=-1)

# Display the image
cv2.imshow('L Shape with Angle', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
