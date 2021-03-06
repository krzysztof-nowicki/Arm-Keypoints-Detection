import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from model import MyAlexNet, ResnetDeco
from skimage import io


class FullResnet(nn.Module):
    def __init__(self, modelA, modelB):
        super(FullResnet, self).__init__()

        self.modelA = modelA
        self.modelB = modelB


    def forward(self, x):
        x = self.modelA(x)
        x = self.modelB(x)
        #print(x.shape)
        out=x
        #print(out)
        return out


net = torchvision.models.resnet34(pretrained=False)

newmodel = torch.nn.Sequential(*(list(net.children())[:-3])) #[1, 1024, 6, 6]
decoder = ResnetDeco()
truenet = FullResnet(newmodel, decoder)

net5 = MyAlexNet()
model = net5

PATH = './bestmodel2.pth'


checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])


while(1):
    imagy = io.imread("37 (Custom).jpg")
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
