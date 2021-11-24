#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import rospy
import sys
import cv2


import rospy
import actionlib
from cv_bridge import CvBridge
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import Image

from model import ResnetDeco
from skimage import io

import glob


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


#FUNCTIONS

#Merging two nn models into one
class MergedModels(nn.Module):
    def __init__(self,  modelA, modelB):
        super(MergedModels, self).__init__()

        self.modelB = modelB
        self.modelA = modelA


    def forward(self, x):
        x = self.modelA(x)
        x = self.modelB(x)
        #print(x.shape)
        out=x
        #print(out)
        return out


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
  

mtx1, dist1 = calibrate_camera(images_folder = 'leftCalib/*')
mtx2, dist2 = calibrate_camera(images_folder = 'rightCalib/*')
 
R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2)

#RT matrix for C1 is identity.
RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
P1 = mtx1 @ RT1 #projection matrix for C1
 
#RT matrix for C2 is the R and T obtained from stereo calibration.
RT2 = np.concatenate([R, T], axis = -1)
P2 = mtx2 @ RT2 #projection matrix for C2 

#KEYPOINTS DETECTION
 
#loading model
rospy.loginfo('Loading model')

estimator = BodyPoseEstimator(pretrained=True)
net = torchvision.models.resnet34(pretrained=False)
newmodel = torch.nn.Sequential(*(list(net.children())[:-3]))
decoder = ResnetDeco()
truenet = MergedModels(newmodel, decoder).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

net4 = torchvision.models.alexnet(pretrained = True)
newmodel3 = torch.nn.Sequential(*(list(net4.children())[:-1]))#[1, 320, 3, 3]
truenet2 = MergedModels(newmodel3, decoder).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


#Selecting the model
model = truenet

rospy.loginfo('Model selected')

#Loading weights 
PATH = 'bigresnetmodel.pth'
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


rospy.loginfo('Model loaded')


#Callback containing image from camera detecting keypoints on given image
def process_image(msg):
    with torch.no_grad():
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(msg, "rgb8")
        img=cv2.resize(img, (96,96))
        img=np.transpose(img, (2, 0, 1)).copy()
        img=torch.tensor(img,dtype=torch.float)
        img = img[None, ...]
        #print(img.shape)
        
        keys = model(img)
        arraykeys=keys.numpy()
        
        
        #wartosci
        arraykeys[0]
        arraykeys[0][0][0]
        arraykeys[0][0][1]
        arraykeys[0][1][0]
        arraykeys[0][1][1]
        arraykeys[0][2][0]
        arraykeys[0][2][1]
        #Calculating the angles
        angle1 = atan2(arraykeys[0][1][1] - arraykeys[0][0][1], arraykeys[0][1][0] - arraykeys[0][0][0]) * 180/pi
        angle2 = atan2(arraykeys[0][2][1] - arraykeys[0][1][1], arraykeys[0][2][0] - arraykeys[0][1][0]) * 180/pi
        angle1degrees=math.degrees(angle1)
        angle2degrees=math.degrees(angle2)
        print(angle1degrees)
        print(angle2degrees)
        #rospy.loginfo('Keypoints detected')
        #print(keys)
    
    
        #drawing an image with keypoints
        
        keys=keys[0].detach().numpy().astype('uint8')
        img=np.transpose(img[0].numpy().astype('uint8'), (1, 2, 0))
        plt.imshow(img)
        plt.scatter(keys[:, 0], keys[:, 1], s=10, marker='x', c='g')
        plt.pause(0.001)
        plt.draw()
        plt.ioff()
        plt.show()
        

    
  
#UR5 CONTROL

#Callback to the state of robotic arm
def process_state(msg):
    
    positions=msg.actual.positions
    c = [0]
    th = np.matrix([[positions[0]], [positions[1]], [positions[2]], [positions[3]], [positions[4]], [positions[5]]])
    P = HTrans(th, c) #Forward kinematics returning matrix 4x4
    #print(P)


[moveA1, moveA2,moveA3,moveA4,moveA5,moveA6]=[2.9, -3.14, 0.0, 0.0, 1.5, 0.0]


class ArmSimpleTrajectory:
    def __init__(self):
    
        global moveA1,moveA2,moveA3,moveA4,moveA5,moveA6
        #rospy.init_node('arm_simple_trajectory')
        
        # Set to True to move back to the starting configurations
        reset = rospy.get_param('~reset', False)
        
        # robot40 ur5 joint names
        arm_joints = ['right_arm_shoulder_pan_joint',
                      'right_arm_shoulder_lift_joint',
                      'right_arm_elbow_joint', 
                      'right_arm_wrist_1_joint',
                      'right_arm_wrist_2_joint',
                      'right_arm_wrist_3_joint']

        
        if reset:
            # Set the arm back to the resting position
            arm_goal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        else:
            # Set a goal configuration for the arm
            arm_goal = [moveA1,moveA2,moveA3,moveA4,moveA5,moveA6]
    
        # Connect to the right arm trajectory action server
        rospy.loginfo('Waiting for ur arm trajectory controller...')
        
        arm_client = actionlib.SimpleActionClient('arm_controller/scaled_pos_joint_traj_controller/follow_joint_trajectory',
                                                  FollowJointTrajectoryAction)
        
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
        
        # moveA2+=0.3
        
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

rospy.init_node('teleop_arm')
if __name__ == '__main__':
    while not rospy.is_shutdown():
    	
        rospy.loginfo('teleop_arm node started')
        
        rospy.Subscriber("/cam1/image_raw", Image, process_image)
        rospy.sleep(1)
        
        rospy.Subscriber("/arm_controller/scaled_pos_joint_traj_controller/state", JointTrajectoryControllerState, process_state)
        ArmSimpleTrajectory()
        itmp=itmp+1
        print("loop number")
        print(itmp)
        print("completed")
