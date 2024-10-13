import numpy as np
import cv2
import torch
from PIL import Image


def add_noise(image, noise_type='gaussian', scale_unifom=50, scale_expo=35, scale_poisson=90, sigma=30,
              salt_pepper_amount=0.05,ksize=(7,7),sigmaX=0):
    
    if noise_type == 'uniform':
        noise = np.random.uniform(low=-scale_unifom, high=scale_unifom, size=image.shape)
    elif noise_type == 'exponential':
        noise = np.random.exponential(scale_expo, size=image.shape)
    elif noise_type == 'blur':
        noisy_image = cv2.GaussianBlur(image, ksize, sigmaX=sigmaX)
        return noisy_image
    elif noise_type == 'poisson':
        noise = np.random.poisson(scale_poisson, size=image.shape)
    elif noise_type == 'gaussian':
        noise = np.random.normal(0, sigma, image.shape)
    elif noise_type == 'salt_pepper':
        noisy_image = np.copy(image)
        # Salt noise
        num_salt = np.ceil(salt_pepper_amount * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[coords[0], coords[1], :] = 255
        # Pepper noise
        num_pepper = np.ceil(salt_pepper_amount * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[coords[0], coords[1], :] = 0
        return noisy_image
    else:
        raise ValueError("Invalid noise type. Choose from 'uniform', 'exponential', 'poisson', 'gaussian', 'salt_pepper'.")

    noisy_image = image + noise
    # Ensure pixel values are within valid range
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image



class AddNoise(torch.nn.Module):
    def __init__(self, **kwargs):
        super(AddNoise, self).__init__()
        self.kwargs = kwargs  # Armazena os kwargs no construtor

    def forward(self, img, noise):
        noisy_image = add_noise(img, noise_type=noise, **self.kwargs)  # Usa os kwargs armazenados no construtor
        return Image.fromarray(noisy_image)

