#!/usr/bin/env python3
import sys

import cv2
from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints


estimator = BodyPoseEstimator(pretrained=True)
while(1):
    image_src = cv2.imread('media/example.jpg')
    keypoints = estimator(image_src)

    # arm
    print(keypoints[0][2][0])  # x
    print(keypoints[0][2][1])  # y
    print(keypoints[0][3])
    print(keypoints[0][4])

#image_dst = draw_body_connections(image_src, keypoints, thickness=4, alpha=0.7)
#image_dst = draw_keypoints(image_dst, keypoints, radius=5, alpha=0.8)
'''
while True:
    cv2.imshow('Image Demo', image_dst)
    if cv2.waitKey(1) & 0xff == 27: # exit if pressed `ESC`
        break
cv2.destroyAllWindows()
'''
