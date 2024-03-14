import cv2
import numpy as np
import math
import random
import geom_util as geom
from roi import ROI


color_dict_HSV = {
    'black': [[180, 255, 30], [0, 0, 0]],
    'white': [[180, 18, 255], [0, 0, 231]],
    'red1': [[180, 255, 255], [159, 50, 70]],
    'red2': [[9, 255, 255], [0, 50, 70]],
    'green': [[89, 255, 255], [36, 50, 70]],
    'blue': [[128, 255, 255], [90, 50, 70]],
    'yellow': [[35, 255, 255], [25, 50, 70]],
    'purple': [[158, 255, 255], [129, 50, 70]],
    'orange': [[24, 255, 255], [10, 50, 70]],
    'gray': [[180, 18, 230], [0, 0, 40]]
}


color_sequence = ['red1', 'red2', 'green', 'blue', 'yellow', 'purple', 'orange']
current_color = color_sequence[0]  # Start with the first color in the sequence


# Load an image from file

imageFrame = cv2.imread('C:/Users/lingz/Downloads/linephotopurple.jpg')  # Make sure to provide the correct path
imageFrame2 = imageFrame.copy()
roi = ROI()
height, width = imageFrame.shape[:2]
roi.init_roi(width, height)  # This sets up the ROI based on your image dimensions

# croppedImage = roi.crop_roi(imageFrame)
# cv2.imshow('Cropped ROI', croppedImage)
mask = np.zeros_like(imageFrame)
top_left = (0, height//2)  # Top-left corner
bottom_right = (width, height)  # Bottom-right corner
# Define the region to keep visible (inverse of hiding). Here, using the whole image for simplicity
cv2.rectangle(mask, top_left, bottom_right, (255, 255, 255), -1)  # White rectangle in the mask

# Apply the mask to the image
# Note: Ensure the mask is in grayscale if the image is in color
croppedImage = cv2.bitwise_and(imageFrame, mask)
cv2.imshow('Masked Image', croppedImage)


hsvFrame = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2HSV)
mask2 = cv2.inRange(imageFrame2, np.array(color_dict_HSV['purple'][1]), 
                   np.array(color_dict_HSV['purple'][0]))

mask = cv2.inRange(hsvFrame, np.array(color_dict_HSV['purple'][1]), 
                   np.array(color_dict_HSV['purple'][0]))
hsvFrame2 = croppedImage.copy()

# # gray = cv2.cvtColor(hsvFrame, cv2.COLOR_BGR2GRAY)
# # cv2.imshow('mask', gray)

# edges = cv2.Canny(mask, 50, 150, apertureSize=3)
# lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=10, minLineLength=90, maxLineGap=0)
# mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
# contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(imageFrame, contours, -1, (255,0, 0), 2)

# cv2.imshow('Contours', imageFrame)

def find_main_countour(image):

    cnts, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    C = cnts
    #take the biggest contour with the biggest area
    if cnts is not None and len(cnts) > 0:
         C = max(cnts, key = cv2.contourArea)


    if C is None:
        return None, None

    #Return coordinates of the box . coordinate top left, top right, bottom right bottom left
    rect = cv2.minAreaRect(C)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box = geom.order_box(box)
    return C, box

def handle_pic(image,image2):
    
    #height and width of input image
    h, w = image.shape[:2]

    cont, box = find_main_countour(image)
    
    #calculates the central axis of the bounding box
    p1, p2 = geom.calc_box_vector(box)
    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))
    print(p1,p2)

    angle = geom.get_vert_angle(p1, p2, w, h)
    shift = geom.get_horz_shift(p1[0], w)

    # draw = fout is not None or show
    draw = True

    if draw:    
        #draw red
        cv2.drawContours(image2, [cont], -1, (0,0,255), 3)
        #draw blue
        cv2.drawContours(image2,[box],0,(255,0,0),2)
        #draw green line
        cv2.line(image2, p1, p2, (0, 255, 0), 3)
        msg_a = "Angle {0}".format(int(angle))
        msg_s = "Shift {0}".format(int(shift))

        cv2.putText(image2, msg_a, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(image2, msg_s, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # if fout is not None:
    #     cv2.imwrite(fout, image2)

    cv2.imshow("image2", image2)
    cv2.waitKey(0)
    return angle, shift

angle = []
# if lines is not None:
#     for line in lines:
#         print("line drawing")
#         x1, y1, x2, y2 = line[0]
#         cv2.line(mask_colored, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         if x2 - x1 == 0:
#             angle += [90]
#         else:
#             angle += [math.atan((y2 - y1) / (x2 - x1)) / np.pi * 180]


handle_pic(mask, imageFrame)
handle_pic(mask, imageFrame2)

print(angle)
print(current_color)

cv2.waitKey(0)  # Change from 1 to 0
cv2.destroyAllWindows()
