import cv2
import numpy as np
import math
import random
import geom_util as geom
from roi import ROI
import time

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

color_sequence = ['red1', 'green', 'blue', 'purple', 'orange']
color_num=0# Start with the first color in the sequence


webcam = cv2.VideoCapture(0) 

def change_color(hsvFrame):
    global color_num
    boo_dict={'red1':True, 'green':True, 'blue':True, 'purple':True, 'orange':True}
    boo_dict[color_sequence[color_num]]=False
    color_dict={0:0,1:0,2:0,3:0,4:0}
    #For red1
    if boo_dict['red1']:
        red1_mask = cv2.inRange(hsvFrame, np.array(color_dict_HSV['red1'][1]),np.array(color_dict_HSV['red1'][0]))
        red1_mask = cv2.dilate(red1_mask, kernel)
        contours, hierarchy = cv2.findContours(red1_mask, 
                                            cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE)
        area = 0
        for pic, contour in enumerate(contours):
            if cv2.contourArea(contour) > area:
                area = cv2.contourArea(contour)
        if(area > 100):
            color_dict[0]=area

    
    #For green
    if boo_dict['green']:
        green_mask = cv2.inRange(hsvFrame, np.array(color_dict_HSV['green'][1]),np.array(color_dict_HSV['green'][0]))
        green_mask = cv2.dilate(green_mask, kernel)
        contours, hierarchy = cv2.findContours(green_mask, 
                                            cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE)
        area = 0
        for pic, contour in enumerate(contours):
            if cv2.contourArea(contour) > area:
                area = cv2.contourArea(contour)
        if(area > 100):
            color_dict[1]=area

    
    #For blue
    if boo_dict['blue']:
        blue_mask = cv2.inRange(hsvFrame, np.array(color_dict_HSV['blue'][1]),np.array(color_dict_HSV['blue'][0]))
        blue_mask = cv2.dilate(blue_mask, kernel)
        contours, hierarchy = cv2.findContours(blue_mask, 
                                            cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE)
        area = 0
        for pic, contour in enumerate(contours):
            if cv2.contourArea(contour) > area:
                area = cv2.contourArea(contour)
        if(area > 100):
            color_dict[2]=area

    #For purple
    if boo_dict['purple']:
        purple_mask = cv2.inRange(hsvFrame, np.array(color_dict_HSV['purple'][1]),np.array(color_dict_HSV['purple'][0]))
        purple_mask = cv2.dilate(purple_mask, kernel)
        contours, hierarchy = cv2.findContours(purple_mask, 
                                            cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE)
        area = 0
        for pic, contour in enumerate(contours):
            if cv2.contourArea(contour) > area:
                area = cv2.contourArea(contour)
        if(area > 100):
            color_dict[3]=area

    #For orange
    if boo_dict['orange']:
        orange_mask = cv2.inRange(hsvFrame, np.array(color_dict_HSV['orange'][1]),np.array(color_dict_HSV['orange'][0]))
        orange_mask = cv2.dilate(orange_mask, kernel)
        contours, hierarchy = cv2.findContours(orange_mask, 
                                            cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE)
        area = 0
        for pic, contour in enumerate(contours):
            if cv2.contourArea(contour) > area:
                area = cv2.contourArea(contour)
        if(area > 100):
            color_dict[4]=area
    max_area=0
    for i in color_dict:
        if color_dict[i]>max_area:
            max_area = color_dict[i]
            max_num=i
    color_num=max_num
            


def find_main_countour(image):

    cnts, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    C = cnts
    #take the biggest contour with the biggest area
    if cnts is not None and len(cnts) > 0:
            C = max(cnts, key = cv2.contourArea)


    if len(C)<=0:
        return None, None
    #Return coordinates of the box . coordinate top left, top right, bottom right bottom left
    if  len(C)>0:
        rect = cv2.minAreaRect(C)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        box = geom.order_box(box)
        return C, box

#image -> image to detect box/lines on. image2-> initial image to draw the box/line on
def handle_pic(image,image2):

    #height and width of input image
    h, w = image.shape[:2]
    cont, box = find_main_countour(image)

    #calculates the central axis of the bounding box
    if box is not None:
        p1, p2 = geom.calc_box_vector(box)
        p1 = (int(p1[0]), int(p1[1]))
        p2 = (int(p2[0]), int(p2[1]))
        #print(p1,p2)

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
        cv2.waitKey(1)
        angle2, shift2 = check_shift_turn(angle, shift)
        turn_state, shift_state = get_turn(angle2, shift2)
        return turn_state, shift_state
    return None,None


def handle_Frame(inputImage, colour):   
    height, width = inputImage.shape[:2]
    mask = np.zeros_like(inputImage)
    top_left = (0, height//2-100)  # Top-left corner
    bottom_right = (width, height)  # Bottom-right corner
    # Define the region to keep visible (inverse of hiding). Here, using the whole image for simplicity
    cv2.rectangle(mask, top_left, bottom_right, (255, 255, 255), -1)  # White rectangle in the mask
    croppedImage = cv2.bitwise_and(inputImage, mask)
    cv2.waitKey(1)  # Change from 1 to 0

    blurredImage = cv2.GaussianBlur(croppedImage, (21, 21), 0)
    hsvFrame = cv2.cvtColor(blurredImage, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsvFrame, np.array(color_dict_HSV[colour][1]),np.array(color_dict_HSV[colour][0]))
    yellow_mask = cv2.dilate(yellow_mask, kernel)
    colourMask = cv2.inRange(hsvFrame, np.array(color_dict_HSV[color_sequence[color_num]][1]), 
                    np.array(color_dict_HSV[color_sequence[color_num]][0]))   
    contours, hierarchy = cv2.findContours(yellow_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE)
    print(color_sequence[color_num])
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour) 
        if(area > 500):
            print('True')
            change_color(hsvFrame)
            time.sleep(3)


    cv2.imshow('yellowmask',yellow_mask)
    return handle_pic(colourMask, imageFrame)

kernel = np.ones((5, 5), "uint8") 

while True:
    _, imageFrame = webcam.read()
   
    print(handle_Frame(imageFrame))
    # Break from loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

