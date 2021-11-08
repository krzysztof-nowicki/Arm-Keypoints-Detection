from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ArmLandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
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

        landmarks = np.array(landmarks, dtype='float32')

        landmarks = landmarks.reshape(-1, 2)

        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            sample['landmarks'] = self.transform(sample['landmarks'])
        return {
            'image': torch.tensor(image, dtype=torch.float),
            'landmarks': torch.tensor(landmarks, dtype=torch.float),
        }
