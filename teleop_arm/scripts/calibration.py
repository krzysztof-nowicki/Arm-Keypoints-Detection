#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import rospy
import sys
import cv2


from cv_bridge import CvBridge
from sensor_msgs.msg import Image


global tmp1, tmp2, tmp3, tmp4
tmp1=0
tmp2=0
tmp3=1
tmp4=1
bridge = CvBridge()
def process_image_cam1(msg):
    global tmp1, tmp3
    img = bridge.imgmsg_to_cv2(msg, "bgr8")
    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img = img[15:495, :]
    
    '''
    h1, w1 = img.shape[:2] 
    print("wymiary kamery:")
    print(w1)
    print(h1)
    print("koniec kamery:")
    '''
    if tmp3 == 1:
        saveLoc1='rightCalib/cali'+str(tmp1)+'.png'
        cv2.imwrite(saveLoc1, img)
        text1="prawe zdjecie "+str(tmp1)+" wrzucone"
        print(text1)
        tmp1=tmp1+1
        tmp3=0

def process_image_cam2(msg):
    global tmp2, tmp4
    img = bridge.imgmsg_to_cv2(msg, "bgr8")
    
    if tmp4 == 1:
        saveLoc2='leftCalib/cali'+str(tmp2)+'.png'
        cv2.imwrite(saveLoc2, img)
        text2="lewe zdjecie "+str(tmp2)+" wrzucone"
        print(text2)
        tmp2=tmp2+1
        tmp4=0


itmp=0
rospy.init_node('calib_sub')
rospy.loginfo('calib_sub node started')
if __name__ == '__main__':


    while not rospy.is_shutdown():
    	
        
        rospy.Subscriber("/cam1/image_raw", Image, process_image_cam1, queue_size=1, buff_size=2**24)
        rospy.Subscriber("/cam2/image_raw", Image, process_image_cam2, queue_size=1, buff_size=2**24)
        input("Press Enter to continue...")
        tmp3=1
        tmp4=1
