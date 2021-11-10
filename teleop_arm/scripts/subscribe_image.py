#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import rospy
import sys
import cv2

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from model import MyAlexNet, ResnetDeco
from skimage import io

class FullResnet(nn.Module):
    def __init__(self,  modelA, modelB):
        super(FullResnet, self).__init__()

        self.modelB = modelB
        self.modelA = modelA


    def forward(self, x):
        x = self.modelA(x)
        x = self.modelB(x)
        #print(x.shape)
        out=x
        #print(out)
        return out


net5 = MyAlexNet()

net = torchvision.models.resnet34(pretrained=False)
newmodel = torch.nn.Sequential(*(list(net.children())[:-3]))
decoder = ResnetDeco()
truenet = FullResnet(newmodel, decoder).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

net4 = torchvision.models.alexnet(pretrained = True)
newmodel3 = torch.nn.Sequential(*(list(net4.children())[:-1]))#[1, 320, 3, 3]
truenet2 = FullResnet(newmodel3, decoder).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


model = truenet



PATH = 'bigresnetmodel.pth'


checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])

def process_image(msg):
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(msg, "rgb8")
    img=cv2.resize(img, (96,96))
    img=np.transpose(img, (2, 0, 1)).copy()
    img=torch.tensor(img,dtype=torch.float)
    img = img[None, ...]
    #print(img.shape)
    keys = model(img)
    print(keys)
   
    keys=keys[0].detach().numpy().astype('uint8')
    img=np.transpose(img[0].numpy().astype('uint8'), (1, 2, 0))
    plt.imshow(img)
    plt.scatter(keys[:, 0], keys[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)
    plt.draw()
    plt.ioff()
    plt.show()
    
    #cv2.imshow("image",img)
    


'''
imagy = io.imread("edge (Custom).jpg")
imagy = np.transpose(imagy, (2, 0, 1)).copy()
imagy = torch.tensor(imagy, dtype=torch.float)
imagy = imagy[None, ...]



keys = model(imagy)
keys=keys[0].detach().numpy().astype('uint8')
imagy=np.transpose(imagy[0].numpy().astype('uint8'), (1, 2, 0))
plt.imshow(imagy)
plt.scatter(keys[:, 0], keys[:, 1], s=10, marker='.', c='r')
plt.pause(0.001)
plt.draw()
plt.ioff()
plt.show()


'''
itmp=0
if __name__ == '__main__':
    while not rospy.is_shutdown():
    	
        rospy.init_node('image_sub')
        rospy.loginfo('image_sub node started')
        rospy.Subscriber("/cam1/image_raw", Image, process_image)
        itmp=itmp+1
        print("odebrano")
        print(itmp)
        print("razy")
        rospy.sleep(1)
        #cv2.imshow("image",image)
        #rospy.spin()
