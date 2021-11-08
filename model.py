import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 5)
        #96-5+1=92
        self.pool1 = nn.MaxPool2d(2,2)
        #92/2=46
        self.conv2 = nn.Conv2d(32, 64, 3)
        #46-3+1=44
        self.pool2 = nn.MaxPool2d(2, 2)
        #44/2=22
        self.conv3 = nn.Conv2d(64, 128, 3)
        #22-3+1=20
        self.pool3 = nn.MaxPool2d(2, 2)
        #20/2=10
        self.conv4 = nn.Conv2d(128, 256, 1)
        # 10-3+1=10
        self.pool4 = nn.MaxPool2d(2, 2)
        #10/2=5
        self.conv5 = nn.Conv2d(256, 512, 2)
        # 5-2+1=4
        self.pool5 = nn.MaxPool2d(2, 2)
        #4/2=2
        self.dropout = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(2, 1024)
        self.fc2 = nn.Linear(1024, 3)
        self.relu = nn.ReLU()
    def forward(self, x):
        #print("początek")
        x = self.conv1(x)
        x = self.relu(x)
        #print(x.shape)
        x = self.pool1(x)
        #print(x.shape)
        x = self.conv2(x)
        x = self.relu(x)
        #rint(x.shape)
        x = self.pool2(x)
        #print(x.shape)
        x = self.conv3(x)
        x = self.relu(x)
        #print(x.shape)
        x = self.pool3(x)
        #print(x.shape)
        x = self.conv4(x)
        x = self.relu(x)
        #print(x.shape)
        x = self.pool4(x)
        #print(x.shape)
        x = self.conv5(x)
        x = self.relu(x)
        #print(x.shape)
        x = self.pool5(x)
        #print(x.shape)
        x = self.fc1(x)
        #print(x.shape)
        x = self.fc2(x)
        #print(x.shape)
        x = x[:,0,:,:]
        #print(x.shape)
        #print("koniec")
        #print(x.shape)
        out=x
        #print(out)
        return out

class MyModel2(nn.Module):
    def __init__(self):
        super(MyModel2, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 1)
        #96-1+1=96
        self.pool1 = nn.MaxPool2d(2,2)
        #96/2=48
        self.conv2 = nn.Conv2d(32, 64, 1)
        #48-1+1=48
        self.pool2 = nn.MaxPool2d(2, 2)
        #48/2=24
        self.conv3 = nn.Conv2d(64, 128, 1)
        #24-1+1=24
        self.pool3 = nn.MaxPool2d(2, 2)
        #24/2=12
        self.conv4 = nn.Conv2d(128, 256, 1)
        # 12-1+1=12
        self.pool4 = nn.MaxPool2d(2, 2)
        #12/2=6
        self.conv5 = nn.Conv2d(256, 512, 1)
        # 6-1+1=6
        self.pool5 = nn.MaxPool2d(2, 2)
        #6/2=3
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(3, 1024)
        self.fc2 = nn.Linear(1024, 2)
    def forward(self, x):
        #print("początek")
        x = self.conv1(x)
        x = self.relu(x)
        #print(x.shape)
        x = self.pool1(x)
        #print(x.shape)
        x = self.conv2(x)
        x = self.relu(x)
        #print(x.shape)
        x = self.pool2(x)
        #print(x.shape)
        x = self.conv3(x)
        x = self.relu(x)
        #print(x.shape)
        x = self.pool3(x)
        #print(x.shape)
        x = self.conv4(x)
        x = self.relu(x)
        #print(x.shape)
        x = self.pool4(x)
        #print(x.shape)
        x = self.conv5(x)
        x = self.relu(x)
        #print(x.shape)
        x = self.pool5(x)
        #print(x.shape)
        x = self.fc1(x)
        #print(x.shape)
        x = self.fc2(x)
        #print(x.shape)
        x = x[:,0,:,:]
        #print(x.shape)
        #print("koniec")
        #print(x.shape)
        out=x
        #print(out)
        return out


class ResnetDeco(nn.Module):
    def __init__(self):
        super(ResnetDeco, self).__init__()

        self.conv1 = nn.Conv2d(256, 512, 4)
        self.fc1 = nn.Linear(3, 1024)
        self.fc2 = nn.Linear(1024, 2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.fc1(x)
        x = self.fc2(x)
        #print(x)
        x = x[:,0,:,:]
        #print(x.shape)
        out=x
        #print(out)
        return out


class EffiDeco(nn.Module):
    def __init__(self):
        super(EffiDeco, self).__init__()

        self.conv1 = nn.Conv2d(1280, 2560, 1)
        self.fc1 = nn.Linear(3, 2560)
        self.fc2 = nn.Linear(2560, 2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.fc1(x)
        x = self.fc2(x)
        #print(x)
        x = x[:,0,:,:]
        #print(x.shape)
        out=x
        #print(out)
        return out

class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
            nn.Conv2d(256, 256, 3, padding=1),  # (b x 256 x  x )
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv1 = nn.Conv2d(256, 512, 1)
        self.fc1 = nn.Linear(3, 512)
        self.fc2 = nn.Linear(512, 2)


    def forward(self, x):
        x = F.interpolate(x, (227,227))
        x = self.net(x)
        x = self.conv1(x)
        x = self.fc1(x)
        x = self.fc2(x)
        #print(x)
        x = x[:,0,:,:]
        #print(x.shape)
        out=x
        #print(out)
        return out



class MergedModels(nn.Module):
    def __init__(self, modelA, modelB):
        super(MergedModels, self).__init__()

        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x):
        x = self.modelA(x)
        x = self.modelB(x)
        # print(x.shape)
        out = x
        # print(out)
        return out
