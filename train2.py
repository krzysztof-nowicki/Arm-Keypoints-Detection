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
from model import MyModel, ResnetDeco, MyModel2, EffiDeco, MyAlexNet
from fastai.vision.all import *
from fastai.metrics import error_rate
#from fastbook import *
from skimage import io


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate = 0.00001
batch_size = 1
Epochs = 200




dataset = ArmLandmarksDataset(csv_file='TemporaryDataset.csv', root_dir='armDataset/', transform=transforms.Compose([
                                               transforms.ToTensor()
                                           ]))

train_set, test_set = torch.utils.data.random_split(dataset, [150, 24])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

#model = MyModel().to(config.DEVICE)








first_conv = nn.Conv2d(1, 3, 1)
net = torchvision.models.resnet34(pretrained=False)
net2 = MyModel2()
net4 = torchvision.models.alexnet(pretrained = True)
net5 = MyAlexNet()
#torch.hub.list('rwightman/gen-efficientnet-pytorch', force_reload=True)
#net3 = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)


#newmodel = torch.nn.Sequential(*(list(net.children())[:-3])) #[1, 1024, 6, 6]
#newmodel2 = torch.nn.Sequential(*(list(net3.children())[:-5]))#[1, 320, 3, 3]
newmodel3 = torch.nn.Sequential(*(list(net4.children())[:-1]))#[1, 320, 3, 3]
decoder = ResnetDeco()
#decoder2 = EffiDeco()

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

#truenet = FullResnet(newmodel, decoder).to(config.DEVICE)
#truenet2 = FullResnet(newmodel2, decoder2).to(config.DEVICE)
truenet3 = FullResnet(newmodel3, decoder).to(config.DEVICE)
#model = FullResnet(newmodel, decoder).to(config.DEVICE)


''' USTAW ODPOWIEDNI MODEL '''
model = net5



optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#criterion = nn.L1Loss()
criterion = nn.MSELoss()


'''
tmp = np.random.rand(96, 96, 1)
tmp = np.transpose(tmp, (2, 0, 1)).copy()
tmp = torch.tensor(tmp, dtype=torch.float)
tmp = first_conv(tmp[None, ...]) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

print(tmp.shape)
tmp = model(tmp)
#tmp = decoder(tmp)
print(tmp.shape)

'''
PATH = './bestmodel2.pth'

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
    if epoch == 199:
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

    #scores2 = scores.view(1, 3, 2)

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
