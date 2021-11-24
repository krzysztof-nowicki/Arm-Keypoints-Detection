#!/usr/bin/env python3

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import rospy
import sys
import cv2
import glob

import rospy
import actionlib
from cv_bridge import CvBridge
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import Image


#import roslib
import tf
#import geometry_msgs.msg
import tf2_msgs.msg
#import turtlesim.srv


from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints

import cmath
import math
from math import cos as cos
from math import sin as sin
from math import atan2 as atan2
from math import acos as acos
from math import asin as asin
from math import sqrt as sqrt
from math import pi as pi


global tmp1, tmp2
global points1
global points2

global points3
global points4

global points5
global points6

global height_limiter

#CALIBRATION FUNCTIONS

def calibrate_camera(images_folder):
    images_names = glob.glob(images_folder)
    images = []
    for imname in images_names:
        im = cv2.imread(imname, 1)
        images.append(im)
 
    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
    rows = 9 #number of checkerboard rows.
    columns = 13 #number of checkerboard columns.
    world_scaling = 1. #change this to the real world square size. Or not.
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
 
    for frame in images:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
        #find the checkerboard
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)
 
        if ret == True:
 
            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)
 
            #opencv can attempt to improve the checkerboard coordinates
            corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, (rows,columns), corners, ret)
            cv2.imshow('img', frame)
            cv2.waitKey(500)
 
            objpoints.append(objp)
            imgpoints.append(corners)
 
 
 
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', mtx)
    print('distortion coeffs:', dist)
    print('Rs:\n', rvecs)
    print('Ts:\n', tvecs)
 
    return mtx, dist
 
def stereo_calibrate(mtx1, dist1, mtx2, dist2):
    #read the synched frames
    c1_images_names = glob.glob('leftCalib/*')
    c1_images_names = sorted(c1_images_names)
    c2_images_names = glob.glob('rightCalib/*')
    c2_images_names = sorted(c2_images_names)
 
    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv2.imread(im1, 1)
        c1_images.append(_im)
 
        _im = cv2.imread(im2, 1)
        c2_images.append(_im)
 
    #change this if stereo calibration not good.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
 
    rows = 9 #number of checkerboard rows.
    columns = 13 #number of checkerboard columns.
    world_scaling = 1. #change this to the real world square size. Or not.
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv2.findChessboardCorners(gray1, (9, 13), None)
        c_ret2, corners2 = cv2.findChessboardCorners(gray2, (9, 13), None)
 
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
 
            cv2.drawChessboardCorners(frame1, (9,13), corners1, c_ret1)
            cv2.imshow('img', frame1)
 
            cv2.drawChessboardCorners(frame2, (9,13), corners2, c_ret2)
            cv2.imshow('img2', frame2)
            cv2.waitKey(500)
 
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
 
    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1, mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
 
    print(ret)
    return R, T

    
def DLT(P1, P2, point1, point2):
 
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    #print('A: ')
    #print(A)
 
    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)
 
    print('Triangulated point: ')
    print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]
  
'''
mtx1, dist1 = calibrate_camera(images_folder = 'leftCalib/*')
mtx2, dist2 = calibrate_camera(images_folder = 'rightCalib/*')
 
R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2)

#RT matrix for C1 is identity.
RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
P1 = mtx1 @ RT1 #projection matrix for C1
 
#RT matrix for C2 is the R and T obtained from stereo calibration.
RT2 = np.concatenate([R, T], axis = -1)
P2 = mtx2 @ RT2 #projection matrix for C2 


'''


#KEYPOINTS DETECTION




 
#loading model
rospy.loginfo('Loading model')

estimator = BodyPoseEstimator(pretrained=True)

rospy.loginfo('Model loaded')


#Callbacks containing images from cameras detecting keypoints on given image
bridge = CvBridge()
def process_image_cam1(msg):
    global tmp1
    img = bridge.imgmsg_to_cv2(msg, "bgr8")
    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img = img[15:495, :]
    
    keypoints = estimator(img)

    global points2, points4, points6
    
    points2=np.array([keypoints[0][2][0],keypoints[0][2][1]])
    points4=np.array([keypoints[0][3][0],keypoints[0][3][1]])
    points6=np.array([keypoints[0][4][0],keypoints[0][4][1]])
    
    print("keypoints:")
    print(keypoints[0][2][0])  # x shoudler
    print(keypoints[0][2][1])  # y
    print(keypoints[0][3][0])  # x elbow
    print(keypoints[0][3][1])  # y
    print(keypoints[0][4][0])  # x wrist
    print(keypoints[0][4][1])  # y
    
    image_dst = draw_body_connections(img, keypoints, thickness=4, alpha=0.7)
    image_dst = draw_keypoints(image_dst, keypoints, radius=5, alpha=0.8)
    image_message = bridge.cv2_to_imgmsg(image_dst, "bgr8")
    image_pub_cam1.publish(image_message)
    tmp1=1

def process_image_cam2(msg):
    global tmp2
    img = bridge.imgmsg_to_cv2(msg, "bgr8")

    keypoints = estimator(img)

    global points1, points3, points5
    
    points1=np.array([keypoints[0][2][0],keypoints[0][2][1]])
    points3=np.array([keypoints[0][3][0],keypoints[0][3][1]])
    points5=np.array([keypoints[0][4][0],keypoints[0][4][1]])
    '''
    print("keypoints:")
    print(keypoints[0][2][0])  # x shoudler
    print(keypoints[0][2][1])  # y
    print(keypoints[0][3][0])  # x elbow
    print(keypoints[0][3][1])  # y
    print(keypoints[0][4][0])  # x wrist
    print(keypoints[0][4][1])  # y
    '''
    image_dst = draw_body_connections(img, keypoints, thickness=4, alpha=0.7)
    image_dst = draw_keypoints(image_dst, keypoints, radius=5, alpha=0.8)
    image_message = bridge.cv2_to_imgmsg(image_dst, "bgr8")
    image_pub_cam2.publish(image_message)
    tmp2=1
    
    
    
    
  
#UR5 CONTROL


[moveA1, moveA2,moveA3,moveA4,moveA5,moveA6]=[2.9, -3.14, 0.0, 0.0, 1.5, 0.0]



class ArmSimpleTrajectory:
    def __init__(self):
    
        global moveA1,moveA2,moveA3,moveA4,moveA5,moveA6,height_limiter
        #moveA1-=0.1
        #moveA3+=0.3
        # UR5 joint names
        arm_joints = ['right_arm_shoulder_pan_joint',
                      'right_arm_shoulder_lift_joint',
                      'right_arm_elbow_joint', 
                      'right_arm_wrist_1_joint',
                      'right_arm_wrist_2_joint',
                      'right_arm_wrist_3_joint']

        if height_limiter<0.9 :
            print(height_limiter)
            print("Danger! Tcp is too close to base! Returning to starting position...")
            [moveA1, moveA2,moveA3,moveA4,moveA5,moveA6]=[2.9, -3.14, -0.6, -1.57, 1.5, -1.57]
        # Set a goal configuration for the arm
        arm_goal = [moveA1,moveA2,moveA3,moveA4,moveA5,moveA6]
    
        # Connect to the right arm trajectory action server
        rospy.loginfo('Waiting for ur arm trajectory controller...')
        
        arm_client = actionlib.SimpleActionClient('arm_controller/scaled_pos_joint_traj_controller/follow_joint_trajectory',  FollowJointTrajectoryAction)
        
        arm_client.wait_for_server()
        
        rospy.loginfo('...connected.')
        rospy.sleep(1)
    
        # Create a single-point arm trajectory with the arm_goal as the end-point
        arm_trajectory = JointTrajectory()
        arm_trajectory.joint_names = arm_joints
        arm_trajectory.points.append(JointTrajectoryPoint())
        arm_trajectory.points[0].positions = arm_goal
        arm_trajectory.points[0].velocities = [0.0 for i in arm_joints]
        arm_trajectory.points[0].accelerations = [0.0 for i in arm_joints]
        arm_trajectory.points[0].time_from_start = rospy.Duration(3)
        
        
        # Send the trajectory to the arm action server
        rospy.loginfo('Moving the arm to goal position...')
        rospy.sleep(1)
        
        # Create an empty trajectory goal
        arm_goal = FollowJointTrajectoryGoal()
        
        # Set the trajectory component to the goal trajectory created above
        arm_goal.trajectory = arm_trajectory
        
        # Specify zero tolerance for the execution time
        arm_goal.goal_time_tolerance = rospy.Duration(0)
    
        # Send the goal to the action server
        arm_client.send_goal(arm_goal)

        rospy.loginfo('...done')
        rospy.sleep(1)



#Main loop
itmp=0

image_pub_cam1 = rospy.Publisher("keypoints_image_cam1",Image, queue_size=1)

image_pub_cam2 = rospy.Publisher("keypoints_image_cam2",Image, queue_size=1)
rospy.init_node('teleop_arm')
if __name__ == '__main__':
    global tmp1, tmp2

    global points1
    global points2
    global points3
    global points4
    global points5
    global points6
    points6=np.array([0, 0])
    points5=np.array([0, 0])
    points1=np.array([0, 0])
    points2=np.array([0, 0])
    points3=np.array([0, 0])
    points4=np.array([0, 0])
    tmp1=0
    tmp2=0
    global height_limiter
    listener = tf.TransformListener()
    listener.waitForTransform('/base_link', '/right_arm_wrist_3_link', rospy.Time(), rospy.Duration(4.0))
    
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            
            (trans,rot) = listener.lookupTransform('/base_link', '/right_arm_wrist_3_link', rospy.Time(0))
            height_limiter=trans[2]
            print(trans)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        

        #rate.sleep()    
    	
        rospy.loginfo('teleop_arm node started')
        
        rospy.Subscriber("/cam1/image_raw", Image, process_image_cam1, queue_size=1, buff_size=2**24)
        #rospy.Subscriber("/cam2/image_raw", Image, process_image_cam2, queue_size=1, buff_size=2**24)
        #rospy.spin()
        '''
        if points1[0]!=0.0 and points1[1]!=0.0 and points2[0]!=0.0 and points2[1]!=0.0 and tmp1!=0 and tmp2!=0 and points3[0]!=0.0 and points3[1]!=0.0 and points4[0]!=0.0 and points4[1]!=0.0 and points5[0]!=0.0 and points5[1]!=0.0 and points6[0]!=0.0 and points6[1]!=0.0:
            print("TRIANGULACJA: ")
            
            print(points1)
            print(points2)
            print(points3)
            print(points4)
            print(points5)
            print(points6)
            
            p_shoulder = DLT(P1, P2, points1, points2)
            p_elbow = DLT(P1, P2, points3, points4)
            p_wrist = DLT(P1, P2, points5, points6)
            
            
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(p_shoulder[0], p_shoulder[1], p_shoulder[2], color='red')
            ax.scatter(p_elbow[0], p_elbow[1], p_elbow[2], color='blue')
            ax.scatter(p_wrist[0], p_wrist[1], p_wrist[2], color='green')
            plt.show()
    
            tmp1=0
            tmp2=0
        '''
        ArmSimpleTrajectory()
        #itmp=itmp+1
        #print("loop number")
        #print(itmp)
        #print("completed")
