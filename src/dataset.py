import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset

import cv2
from PIL import Image
import pandas as pd



class CustomImageDataset(Dataset):
    def __init__(self, dataframe:pd.DataFrame, transform:v2.Compose, noise_transform:v2.Compose):
        self.dataframe = dataframe
        self.transform = transform
        self.noise_transform = noise_transform

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):

        path,_,_,noise,_ = self.dataframe.iloc[idx]

        img = cv2.imread(path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        noise_img = img.copy()
        noise_img = self.noise_transform(noise_img,noise)

        img = Image.fromarray(img)
        img = self.transform(img)
        noise_img = self.transform(noise_img)

        return img, noise_img
