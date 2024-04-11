import cv2
import numpy as np
import math
import random
import geom_util as geom
from roi import ROI

# line angle to make a turn
turn_angle = 45
# line shift to make an adjustment
shift_max = 20
# turning time of turn
turn_step = 0.25
# turning time of shift adjustment
shift_step = 0.125


def check_shift_turn(angle, shift):
    turn_state = 0
    #if angle difference bigger than turn_angle then return the angle direction to turn
    #if angle is 45(pointing to the right) returns positive
    if angle < turn_angle or angle > 180 - turn_angle:
        turn_state = np.sign(90 - angle)

    shift_state = 0
    #if horizonal shift bigger than allowed, return whether the line is too far to the left 
    #or right side of the robot.
    # if line is to the left , return -1. if line to right return 1. 
    if abs(shift) > shift_max:
        shift_state = np.sign(shift)
    return turn_state, shift_state

def get_turn(turn_state, shift_state):
    turn_dir = 0
    turn_val = 0
    if shift_state != 0:
        turn_dir = shift_state
        turn_val = shift_step if shift_state != turn_state else turn_step
    elif turn_state != 0:
        turn_dir = turn_state
        turn_val = turn_step
    #turn value -> how long to turn the robot for.
    return turn_dir, turn_val   

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
current_color = color_sequence[2]  # Start with the first color in the sequence

imageFrame = cv2.imread('C:/Users/lingz/Downloads/photo2.jpg')  # Make sure to provide the correct path


height, width = imageFrame.shape[:2]
mask = np.zeros_like(imageFrame)
top_left = (0, height//2-100)  # Top-left corner
# bottom_right = (width, 2*height//4)  # Bottom-right corner
bottom_right = (width, height)  # Bottom-right corner

# Define the region to keep visible (inverse of hiding). Here, using the whole image for simplicity
cv2.rectangle(mask, top_left, bottom_right, (255, 255, 255), -1)  # White rectangle in the mask
croppedImage = cv2.bitwise_and(imageFrame, mask)
blurredImage = cv2.GaussianBlur(croppedImage, (21, 21), 0.5)

cv2.waitKey(0)  # Change from 1 to 0

hsvFrame = cv2.cvtColor(blurredImage, cv2.COLOR_BGR2HSV)
colourMask = cv2.inRange(hsvFrame, np.array(color_dict_HSV['green'][1]), 
                   np.array(color_dict_HSV['green'][0]))

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

#image -> image to detect box/lines on. image2-> initial image to draw the box/line on
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
        #draw 
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
    angle2, shift2 = check_shift_turn(angle, shift)
    turn_state, shift_state = get_turn(angle2, shift2)
    return turn_state, shift_state



print(handle_pic(colourMask, imageFrame))
key = cv2.waitKey(0)  # Waits indefinitely until a key is pressed
if key == 27:  # 27 is the ASCII code for the ESC key
    cv2.destroyAllWindows()
