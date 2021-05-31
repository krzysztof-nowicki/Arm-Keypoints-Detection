import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import utils
import config

from model import FaceKeypointModel
from tqdm import tqdm

# image resize dimension
resize = 96

model = FaceKeypointModel().to(config.DEVICE)
# load the model checkpoint
checkpoint = torch.load(f"{config.OUTPUT_PATH}/model.pth")
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# read the test CSV file
csv_file = f"{config.ROOT_PATH}/test/test.csv"
data = pd.read_csv(csv_file)
pixel_col = data.Image
image_pixels = []
for i in tqdm(range(len(pixel_col))):
    img = pixel_col[i].split(' ')
    image_pixels.append(img)
# convert to NumPy array
images = np.array(image_pixels, dtype='float32')

images_list, outputs_list = [], []
for i in range(9):
    with torch.no_grad():
        image = images[i]
        image = image.reshape(96, 96, 1)
        image = cv2.resize(image, (resize, resize))
        image = image.reshape(resize, resize, 1)
        orig_image = image.copy()
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float)
        image = image.unsqueeze(0).to(config.DEVICE)

        # forward pass through the model
        outputs = model(image)
        # append the current original image
        images_list.append(orig_image)
        # append the current outputs
        outputs_list.append(outputs)
utils.test_keypoints_plot(images_list, outputs_list)