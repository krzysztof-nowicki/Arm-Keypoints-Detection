import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import config

from torch.utils.data import DataLoader
from dataset import KeyPointsDataset
from model import FaceKeypointModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate = 0.0001
batch_size = 1
Epochs = 300


dataset = KeyPointsDataset(csv_file='armDataBackup.csv', root_dir='armDataset')

train_set, test_set = torch.utils.data.random_split(dataset, [33, 7])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

model = FaceKeypointModel().to(config.DEVICE)

optimizer = optim.Adam(model.parameters(), lr=config.LR)

criterion = nn.MSELoss()

