import numpy as np
import pandas as pd
#import tensorflow as tf
import matplotlib
import os
import fastai
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image
from fastai.vision.all import *
from fastai.vision import *
from pathlib import Path
from skimage import io, transform
from datasetloading import ArmLandmarksDataset, show_landmarks

'''
image = io.imread("harry (Custom).jpg")
print(image.shape)
image2 = np.transpose(image, (2, 0, 1)).copy()
print(image2.shape)
image3= torch.tensor(image2, dtype=torch.float)
print(image3.shape)
df = pd.read_csv('armDataset/armDataBackup.csv')
#print(df.head())

dls = ImageDataLoaders.from_csv('./armDataset', 'armDataBackup.csv', num_workers=0)



learn = cnn_learner(dls, resnet50)


learn.fit_one_cycle(4)

preds = learn.predict(image)
print(preds)
'''

df = pd.read_csv('armDataset/armDataBackup.csv')
print(df.head())

dataset = ArmLandmarksDataset(csv_file='armDataBackup.csv', root_dir='armDataset/', transform=transforms.Compose([
                                               transforms.ToTensor()
                                           ]))
lands = dataset[0]['landmarks']
mages = dataset[0]['image']

labls=dataset['landmarks']
imagesdata=dataset['image']
image=np.transpose(mages.numpy().astype('uint8'), (1, 2, 0))
img = PILImage.create(image)

def get_output(f):
    keyp = dataset[f]['landmarks']
    return keyp
tes = get_output(0)
print(tes)

item_tfms = [Resize(96, method='squish')]
batch_tfms = [Flip(), Rotate(), Zoom(), Warp()]


dblock = DataBlock(blocks=(ImageBlock, PointBlock),
                   get_items=get_image_files,
                   splitter=RandomSplitter(),
                   get_y=get_output,
                   item_tfms=item_tfms,
                   batch_tfms=batch_tfms)
dblock.summary('')