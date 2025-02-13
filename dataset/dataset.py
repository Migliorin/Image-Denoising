import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset

import cv2
from PIL import Image
import numpy as np

from basicsr.data import degradations as degradations
import math
from random import randint

import os


class FFHQDegradationDataset(Dataset):
    def __init__(self, ffhq_path:str, transform:v2.Compose, noise_transform:v2.Compose,
                 img_shape=(512,512)):
        self.paths = self._get_images_path(ffhq_path)
        self.transform = transform
        self.noise_transform = noise_transform
        self.img_shape = img_shape

        # degradation configurations
        self.blur_kernel_size = 41
        self.kernel_list = ['iso','aniso']
        self.kernel_prob = [0.5, 0.5]
        self.blur_sigma = [0.1, 10]
        self.downsample_range = [0.8, 8]
        self.noise_range = [0, 20]
        self.jpeg_range = [60, 100]


    def _get_images_path(self, path:str) -> str:
        aux = []
        for root_, folder_, files_ in os.walk(path):
            if(len(files_)):
                for file_ in files_:
                    if(file_.endswith(("png","jpg","jpeg"))):
                        aux.append(f"{root_}/{file_}")
        return aux

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        path = self.paths[idx]

        img_gt = cv2.imread(path)
        img_gt = cv2.cvtColor(img_gt,cv2.COLOR_BGR2RGB)

        img_gt = (img_gt/255).astype(np.float32)
        h, w, _ = img_gt.shape

        kernel = degradations.random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            self.blur_sigma,
            self.blur_sigma, [-math.pi, math.pi],
            noise_range=None)

        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)

        # noise
        if self.noise_range is not None:
            img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:

            img_lq = degradations.add_jpg_compression(
                img_lq,
                quality=randint(self.jpeg_range[0],self.jpeg_range[1])
            )

        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        img_gt = cv2.resize(img_gt,self.img_shape)
        img_lq = cv2.resize(img_lq,self.img_shape)

        img_gt = np.transpose(img_gt,(2,0,1))
        img_lq = np.transpose(img_lq,(2,0,1))

        img_gt = self.transform(img_gt)
        img_lq = self.noise_transform(img_lq)

        return img_gt, img_lq

