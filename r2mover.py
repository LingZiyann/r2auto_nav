# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# adapted from https://github.com/Shashika007/teleop_twist_keyboard_ros2/blob/foxy/teleop_twist_keyboard_trio/teleop_keyboard.py

import rclpy
from rclpy.node import Node
import geometry_msgs.msg
import cv2
import numpy as np
import math
import geom_util as geom
from roi import ROI
import time
from HTTP import HTTP_requests


# constants
rotatechange = 0.1
speedchange = 0.05

# line angle to make a turn
turn_angle = 45
# line shift to make an adjustment
shift_max = 20
# turning time of turn
turn_step = 0.25
# turning time of shift adjustment
shift_step = 0.125
httpSent = False
Door = ''

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

webcam = cv2.VideoCapture(0) 
color_sequence = ['red1', 'green', 'blue', 'purple']
color_num=2# Start with the first color in the sequence

kernel = np.ones((5, 5), "uint8") 


def change_color(hsvFrame):
    global color_num
    boo_dict={'red1':True, 'green':True, 'blue':True, 'purple':True}
    boo_dict[color_sequence[color_num]]=False
    color_dict={0:0,1:0,2:0,3:0}
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
    # if boo_dict['orange']:
    #     orange_mask = cv2.inRange(hsvFrame, np.array(color_dict_HSV['orange'][1]),np.array(color_dict_HSV['orange'][0]))
    #     orange_mask = cv2.dilate(orange_mask, kernel)
    #     contours, hierarchy = cv2.findContours(orange_mask, 
    #                                         cv2.RETR_TREE, 
    #                                         cv2.CHAIN_APPROX_SIMPLE)
    #     area = 0
    #     for pic, contour in enumerate(contours):
    #         if cv2.contourArea(contour) > area:
    #             area = cv2.contourArea(contour)
    #     if(area > 100):
    #         color_dict[4]=area
    max_area=0
    for i in color_dict:
        if color_dict[i]>max_area:
            max_area = color_dict[i]
            max_num=i
    color_num=max_num

def find_main_countour(image):

    cnts, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(cnts)
    C = cnts
    #take the biggest contour with the biggest area
    if cnts is not None and len(cnts) > 0:
            # print('1')
            C = max(cnts, key = cv2.contourArea)


    if len(C)<=0:
        print('2')
        return None, None
    # print(C) 
    #Return coordinates of the box . coordinate top left, top right, bottom right bottom left
    if  len(C)>0:
        rect = cv2.minAreaRect(C)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        box = geom.order_box(box)
        return C, box

def handle_Frame(inputImage):  
    global boo   
    global Door 
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
    yellow_mask = cv2.inRange(hsvFrame, np.array(color_dict_HSV['yellow'][1]),np.array(color_dict_HSV['yellow'][0]))
    yellow_mask = cv2.dilate(yellow_mask, kernel)
    colourMask = cv2.inRange(hsvFrame, np.array(color_dict_HSV[color_sequence[color_num]][1]), 
                    np.array(color_dict_HSV[color_sequence[color_num]][0]))   
    contours_yellow, hierarchy_yellow = cv2.findContours(yellow_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE)
    orange_mask = cv2.inRange(hsvFrame, np.array(color_dict_HSV['orange'][1]),np.array(color_dict_HSV['orange'][0]))
    orange_mask = cv2.dilate(orange_mask, kernel)
    print(color_sequence[color_num])
    contours_orange, hierarchy_orange = cv2.findContours(orange_mask, 
                                        cv2.RETR_TREE, 
                                        cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours_yellow):
        area = cv2.contourArea(contour) 
        if(area > 500):
            if not boo:
                print('True, change color')
                change_color(hsvFrame)
                time.sleep(3)
                break
            #else:
                #To SHOOT

    for pic, contour in enumerate(contours_orange):
        area = cv2.contourArea(contour) 
        if(area > 500):
            print('True, start HTTP')
            Door = HTTP_requests()
            while Door !='Door 1' or Door != 'Door 2':
                Door = HTTP_requests()
                
            else:
                boo = True 
                time.sleep(3)
                break

    cv2.imshow('yellowmask',yellow_mask)
    return handle_pic(colourMask, inputImage)

#image -> image to detect box/lines on. image2-> initial image to draw the box/line on
def handle_pic(image,image2):

    #height and width of input image
    h, w = image.shape[:2]
    # print(image)
    cont, box = find_main_countour(image)

    #calculates the central axis of the bounding box
    if box is not None:
        p1, p2 = geom.calc_box_vector(box)
        p1 = (int(p1[0]), int(p1[1]))
        p2 = (int(p2[0]), int(p2[1]))
        # print(p1,p2)

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
        angle2, shift2 = check_shift_turn(angle, shift)
        turn_state, shift_state = get_turn(angle2, shift2)
        return turn_state, shift_state
    return None,None



class Mover(Node):
    def __init__(self):
        super().__init__('mover')
        self.publisher_ = self.create_publisher(geometry_msgs.msg.Twist, 'cmd_vel', 10)
        self.move_timer = None


    def detect_color_and_move(self, turn_state, shift_timer):
        start_time = time.time() 
        print(turn_state, shift_timer)
        #turn-state -> which direction to turn. shift timer -> how long to turn for. 
        twist = geometry_msgs.msg.Twist()
        if turn_state == -1:
            # Move backward
            twist.linear.x = speedchange
            twist.angular.z = rotatechange
        elif turn_state == 1:
            # Move forward
            twist.linear.x = speedchange
            twist.angular.z = rotatechange

        if self.move_timer is not None:
            self.move_timer.cancel()

        self.publisher_.publish(twist)
        #notsure if its syncrhonous or asynchronous
        self.move_timer = self.create_timer(shift_timer, self.stop_movement)

        elapsed_time = time.time() - start_time + shift_timer # Calculate elapsed time

        return elapsed_time
    
    def stop_movement(self):
        # Stop the robot by publishing a zero-velocity Twist message
        twist = geometry_msgs.msg.Twist()
        twist.linear.z = 0
        self.publisher_.publish(twist)
        if self.move_timer is not None:
            self.move_timer.cancel() 

def main(args=None):
    global color_num
    rclpy.init(args=args)

    mover = Mover()
    webcam = cv2.VideoCapture(0) 
    twist = geometry_msgs.msg.Twist()

    try:
        while rclpy.ok():
            _, imageFrame = webcam.read() 
            if httpSent:
                if Door == 'Door 1':
                    #follow green
                    color_num = 1
                else:
                    #follow red
                    color_num = 0

            turn_state, shift_state = handle_Frame(imageFrame)
            if (turn_state != None):
                print("moving")
                twist.linear.x = speedchange
                mover.publisher_.publish(twist)

            if (turn_state != 0):
                time_taken = mover.detect_color_and_move(turn_state, shift_state)
            else:
                ret = webcam.grab()

    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        mover.webcam.release()
        mover.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
