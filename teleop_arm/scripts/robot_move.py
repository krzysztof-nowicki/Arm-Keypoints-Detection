#!/usr/bin/env python3


import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# [2.9000264312563373, -1.899919021949695, -1.7500820091534157, 6.470402649938478e-05, 1.5000068492037473, 0.7499506414304014]

[moveA1, moveA2,moveA3,moveA4,moveA5,moveA6]=[2.9, -3.14, 0.0, 0.0, 1.5, 0.0]

import numpy as np
from numpy import linalg


import cmath
import math
from math import cos as cos
from math import sin as sin
from math import atan2 as atan2
from math import acos as acos
from math import asin as asin
from math import sqrt as sqrt
from math import pi as pi

global mat
mat=np.matrix




global d, a, alph
#source https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
d = mat([0.089159, 0, 0, 0.10915, 0.09465, 0.0823])
a =mat([0 ,-0.425 ,-0.39225 ,0 ,0 ,0])
alph = mat([math.pi/2, 0, 0, math.pi/2, -math.pi/2, 0 ])
# ************************************************** FORWARD KINEMATICS

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

def process_state(msg):
    
    positions=msg.actual.positions
    c = [0]
    th = np.matrix([[positions[0]], [positions[1]], [positions[2]], [positions[3]], [positions[4]], [positions[5]]])
    P = HTrans(th, c)
    print(P)


class ArmSimpleTrajectory:
    def __init__(self):
    
        global moveA1,moveA2,moveA3,moveA4,moveA5,moveA6
        rospy.init_node('arm_simple_trajectory')
        
        # Set to True to move back to the starting configurations
        reset = rospy.get_param('~reset', False)
        
        # Which joints define the arm?
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


if __name__ == '__main__':
    
    
    try:
        while 1 :
            rospy.Subscriber("/arm_controller/scaled_pos_joint_traj_controller/state", JointTrajectoryControllerState, process_state)
            ArmSimpleTrajectory()
    except rospy.ROSInterruptException:
        pass
