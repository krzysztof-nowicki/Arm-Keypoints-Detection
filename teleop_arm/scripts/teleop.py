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

#imports used for forward kinematics
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

#Forward kinematics
global mat
mat=np.matrix

global d, a, alph
#source https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
d = mat([0.089159, 0, 0, 0.10915, 0.09465, 0.0823])
a =mat([0 ,-0.425 ,-0.39225 ,0 ,0 ,0])
alph = mat([math.pi/2, 0, 0, math.pi/2, -math.pi/2, 0 ])

def AH( n,th,c  ):

  T_a = mat(np.identity(4), copy=False)
  T_a[0,3] = a[0,n-1]
  T_d = mat(np.identity(4), copy=False)
  T_d[2,3] = d[0,n-1]

  Rzt = mat([[cos(th[n-1,c]), -sin(th[n-1,c]), 0 ,0],
	         [sin(th[n-1,c]),  cos(th[n-1,c]), 0, 0],
	         [0,               0,              1, 0],
	         [0,               0,              0, 1]],copy=False)
      

  Rxa = mat([[1, 0,                 0,                  0],
			 [0, cos(alph[0,n-1]), -sin(alph[0,n-1]),   0],
			 [0, sin(alph[0,n-1]),  cos(alph[0,n-1]),   0],
			 [0, 0,                 0,                  1]],copy=False)

  A_i = T_d * Rzt * T_a * Rxa
	    

  return A_i

def HTrans(th,c ):  
  A_1=AH( 1,th,c  )
  A_2=AH( 2,th,c  )
  A_3=AH( 3,th,c  )
  A_4=AH( 4,th,c  )
  A_5=AH( 5,th,c  )
  A_6=AH( 6,th,c  )
      
  T_06=A_1*A_2*A_3*A_4*A_5*A_6

  return T_06
  
  
  
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


#KEYPOINTS DETECTION
  
#loading models
rospy.loginfo('Loading models')

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
        print(arraykeys[0])
        print(arraykeys[0][0][0])
        print(arraykeys[0][0][1])
        print(arraykeys[0][1][0])
        print(arraykeys[0][1][1])
        print(arraykeys[0][2][0])
        print(arraykeys[0][2][1])
        

        #rospy.loginfo('Keypoints detected')
        #print(keys)
    
    
        #drawing an image with keypoints
        '''
        keys=keys[0].detach().numpy().astype('uint8')
        img=np.transpose(img[0].numpy().astype('uint8'), (1, 2, 0))
        plt.imshow(img)
        plt.scatter(keys[:, 0], keys[:, 1], s=10, marker='x', c='g')
        plt.pause(0.001)
        plt.draw()
        plt.ioff()
        plt.show()
        '''

    
  
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
