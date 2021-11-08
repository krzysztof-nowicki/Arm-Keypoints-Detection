import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import config
import numpy as np
import matplotlib.pyplot as plt
import cv2

from torch.utils.data import DataLoader
from datasetloading import ArmLandmarksDataset
from model import MyModel, ResnetDeco, MyModel2, EffiDeco, MyAlexNet, MergedModels
from fastai.vision.all import *
from skimage import io

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

#SET ALL PARAMETERS
learning_rate = 0.0001
batch_size = 1
Epochs = 100
dataset_name ="smaller_mpii_mine_armdataset.csv"
dataset_root_dir ="armDataset/"
dataset_split = [1000, 20, 98] #Has to be accurate to numbers in csv file
PATH = './mysmallerresnetmodel.pth' #Name for saving weights of a model


#Dataset loading
dataset = ArmLandmarksDataset(csv_file=dataset_name, root_dir=dataset_root_dir, transform=transforms.Compose([
    transforms.ToTensor()
]))
train_set, test_set, valid_set = torch.utils.data.random_split(dataset, dataset_split)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True)



#Few different backbone models to test out
net = torchvision.models.resnet34(pretrained=True)
net2 = MyModel2()
net4 = torchvision.models.alexnet(pretrained = True)
net5 = MyAlexNet()
net6 = torchvision.models.mobilenet_v2(pretrained = True)
#torch.hub.list('rwightman/gen-efficientnet-pytorch', force_reload=True)
#net3 = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)



#Last few layers of different models are cut in order to change the output of a model
newmodel = torch.nn.Sequential(*(list(net.children())[:-3]))  # [1, 1024, 6, 6]
#newmodel2 = torch.nn.Sequential(*(list(net3.children())[:-5]))#[1, 320, 3, 3]
newmodel3 = torch.nn.Sequential(*(list(net4.children())[:-1]))#[1, 320, 3, 3]
newmodel4 = torch.nn.Sequential(*(list(net6.children())[:-1]))#[1, 1280, 3, 3]


#Declaration of decoders of previously cut models defined in model.py file
decoder = ResnetDeco()
decoder2 = EffiDeco()


#Cut models and decoders are merged
truenet = MergedModels(newmodel, decoder)
#truenet2 = FullResnet(newmodel2, decoder2).to(config.DEVICE)
truenet3 = MergedModels(newmodel3, decoder)
# model = FullResnet(newmodel, decoder).to(config.DEVICE)
truenet4 = MergedModels(newmodel4, decoder2)

#The choice of a model from merged above
model = truenet.to(config.DEVICE)


#Parameters for training
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# criterion = nn.L1Loss()
criterion = nn.MSELoss()


#Simple code to test out if models are working properly
'''
tmp = np.random.rand(96, 96, 1)
tmp = np.transpose(tmp, (2, 0, 1)).copy()
tmp = torch.tensor(tmp, dtype=torch.float)
tmp = first_conv(tmp[None, ...]) 

print(tmp.shape)
tmp = truenet4(tmp)
#tmp = decoder(tmp)
print(tmp.shape)
'''



for epoch in range(Epochs):
    losses = []
    valid_losses = []
    for i, data in enumerate(train_loader):
        image = data['image'].to(device=device)
        keypoints = data['landmarks'].to(device=device)

        scores = model(image)

        # keypoints = keypoints.view(1,2,3)
        # print(scores2.shape)
        # print(scores2)
        # print(keypoints.shape)
        # print(keypoints)
        # print(keypoints.shape)
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
        loss = criterion(scores, keypoints)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


    print(f'Epoch: {epoch}, Cost: {sum(losses) / len(losses)}')


    for k, valid_data in enumerate(valid_loader):
        valid_image = valid_data['image'].to(device=device)
        valid_keypoints = valid_data['landmarks'].to(device=device)
        valid_scores = model(valid_image)
        valid_loss = criterion(valid_scores, valid_keypoints)
        valid_losses.append(valid_loss.item())

    print(f'Valid loss: {sum(valid_losses) / len(valid_losses)}')

    if epoch == Epochs-1:
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
keys = keys[0].detach().numpy().astype('uint8')
imagy = np.transpose(imagy[0].numpy().astype('uint8'), (1, 2, 0))
plt.imshow(imagy)
plt.scatter(keys[:, 0], keys[:, 1], s=10, marker='.', c='r')
plt.pause(0.001)
plt.draw()
plt.ioff()
plt.show()

for j, data in enumerate(test_loader):
    image = data['image']
    plt.figure()
    # scores = model(image)

    scores = model(image)
    # image = image[0, :, :, :]

    # scores2 = scores.view(1, 3, 2)

    print(scores)
    print(data['landmarks'])
    scores = scores[0].detach().numpy().astype('uint8')
    image = np.transpose(image[0].numpy().astype('uint8'), (1, 2, 0))
    plt.imshow(image)
    plt.scatter(scores[:, 0], scores[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)
    plt.draw()
    plt.ioff()
    plt.show()
    # time.sleep(5)
    # show_landmarks(image[0], scores)
