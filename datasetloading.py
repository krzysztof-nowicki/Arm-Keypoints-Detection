from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


plt.ion()


landmarks_frame = pd.read_csv('armDataset/armDataBackup.csv')

n = 15
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:]
landmarks = np.asarray(landmarks)
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 3 Landmarks: {}'.format(landmarks[:3]))

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

#plt.figure()
#show_landmarks(io.imread(os.path.join('armDataset/', img_name)),
#               landmarks)
#plt.show()


class ArmLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        image = np.transpose(image, (2, 0, 1)).copy()


        landmarks = self.landmarks_frame.iloc[idx, 1:]
        #landmarks = np.array([landmarks])
        landmarks = np.array(landmarks, dtype='float32')
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        landmarks = landmarks.reshape(-1, 2)

        sample = {'image': image , 'landmarks': landmarks}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            sample['landmarks'] = self.transform(sample['landmarks'])
            #print(sample['image'])
            #print(sample['landmarks'])
        #return sample
        return {
            'image': torch.tensor(image, dtype=torch.float),
            'landmarks': torch.tensor(landmarks, dtype=torch.float),
        }


        # get the keypoints


        # reshape the keypoints

        # rescale keypoints according to image resize



arm_dataset = ArmLandmarksDataset(csv_file='armDataset/armDataBackup.csv',
                                    root_dir='armDataset/', transform=transforms.Compose([
                                               transforms.ToTensor()
                                           ]))
"""
fig = plt.figure()

for i in range(len(arm_dataset)):
    sample = arm_dataset[i]

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break
"""
