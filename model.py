import torch.nn as nn
import torch.nn.functional as F


class FaceKeypointModel(nn.Module):
    def __init__(self):
        super(FaceKeypointModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(p=0.2)
    def forward(self, x):

         x = F.relu(self.conv1(x))

         x = self.pool(x)

         x = F.relu(self.conv2(x))

         x = self.pool(x)

         x = F.relu(self.conv3(x))

         x = self.pool(x)

         bs, _, _, _ = x.shape

         x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)

         x = self.dropout(x)

         out = self.fc1(x)

         return out

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
    def forward(self, x):
        print("początek")
        x = self.conv1(x)
        print(x.shape)
        x = self.pool1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.pool2(x)
        print(x.shape)
        x = self.conv3(x)
        print(x.shape)
        x = self.pool3(x)
        print(x.shape)
        x = self.conv4(x)
        print(x.shape)
        x = self.pool4(x)
        print(x.shape)
        x = self.conv5(x)
        print(x.shape)
        x = self.pool5(x)
        print(x.shape)
        x = self.fc1(x)
        print(x.shape)
        x = self.fc2(x)
        print(x.shape)
        x = x[:,0,:,:]
        print(x.shape)
        print("koniec")
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

        self.dropout = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(3, 1024)
        self.fc2 = nn.Linear(1024, 2)
    def forward(self, x):
        print("początek")
        x = self.conv1(x)
        print(x.shape)
        x = self.pool1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.pool2(x)
        print(x.shape)
        x = self.conv3(x)
        print(x.shape)
        x = self.pool3(x)
        print(x.shape)
        x = self.conv4(x)
        print(x.shape)
        x = self.pool4(x)
        print(x.shape)
        x = self.conv5(x)
        print(x.shape)
        x = self.pool5(x)
        print(x.shape)
        x = self.fc1(x)
        print(x.shape)
        x = self.fc2(x)
        print(x.shape)
        x = x[:,0,:,:]
        print(x.shape)
        print("koniec")
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
        x = x[:,0,:,:]
        #print(x.shape)
        out=x
        #print(out)
        return out