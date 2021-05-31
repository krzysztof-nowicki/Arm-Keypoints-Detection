import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import config
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time




from torch.utils.data import DataLoader
from datasetloading import ArmLandmarksDataset, show_landmarks
from model import MyModel, ResnetDeco, MyModel2
from fastai.vision.all import *
from fastai.metrics import error_rate
#from fastbook import *
from skimage import io


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate = 0.0001
batch_size = 1
Epochs = 250




dataset = ArmLandmarksDataset(csv_file='armDataBackup.csv', root_dir='armDataset/', transform=transforms.Compose([
                                               transforms.ToTensor()
                                           ]))

train_set, test_set = torch.utils.data.random_split(dataset, [33, 7])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

model = MyModel2().to(config.DEVICE)








first_conv = nn.Conv2d(1, 3, 1)
net = torchvision.models.resnet34(pretrained=True)



newmodel = torch.nn.Sequential(*(list(net.children())[:-3])) #[1, 1024, 6, 6]
decoder = ResnetDeco()

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

truenet = FullResnet(newmodel, decoder).to(config.DEVICE)


optimizer = optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.MSELoss()

'''
tmp = np.random.rand(96, 96, 1)
tmp = np.transpose(tmp, (2, 0, 1)).copy()
tmp = torch.tensor(tmp, dtype=torch.float)
tmp = first_conv(tmp[None, ...]) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

print(tmp.shape)
tmp = newmodel(tmp)
#tmp = decoder(tmp)
print(tmp.shape)
'''

PATH = './mojmodel3.pth'

for epoch in range(Epochs):
    losses = []

    for i, data in enumerate(train_loader):
        image = data['image'].to(device=device)
        keypoints = data['landmarks'].to(device=device)
        #scores = model(image)
        #image = first_conv(image[None, ...])

        scores = model(image)



        #keypoints = keypoints.view(1,2,3)
        #print(scores2.shape)
        #print(scores2)
        #print(keypoints.shape)
        #print(keypoints)
        #print(keypoints.shape)
        """
        print(keypoints)
        print(scores)
        image = np.transpose(image[0].numpy().astype('uint8'), (1, 2, 0))
        plt.imshow(image)
        plt.pause(0.001)
        plt.draw()
        plt.ioff()
        plt.show()
        """
        loss=criterion(scores, keypoints)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    print(f'Epoch: {epoch}, Cost: {sum(losses)/len(losses)}')
    if epoch == 249:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss, }, PATH)




checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.eval()


imagy = io.imread("harry (Custom).jpg")
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

for j, data in enumerate(test_loader):

    image = data['image']
    plt.figure()
    #scores = model(image)

    scores = model(image)
    #image = image[0, :, :, :]

    scores2 = scores.view(1, 3, 2)

    print(scores)
    print(data['landmarks'])
    scores=scores[0].detach().numpy().astype('uint8')
    image=np.transpose(image[0].numpy().astype('uint8'), (1, 2, 0))
    plt.imshow(image)
    plt.scatter(scores[:, 0], scores[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)
    plt.draw()
    plt.ioff()
    plt.show()
    #time.sleep(5)
    #show_landmarks(image[0], scores)



