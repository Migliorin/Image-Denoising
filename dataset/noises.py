import numpy as np
import cv2
import torch
from PIL import Image
import random

def apply_gaussian_blur(image, sigma):
    # Aplicar desfoque gaussiano
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def downsample_image(image, factor):
    # Reduzir a resolução da imagem
    return cv2.resize(image, (image.shape[1] // factor, image.shape[0] // factor), interpolation=cv2.INTER_LINEAR)

def add_gaussian_noise(image, delta):
    # Adicionar ruído gaussiano
    row, col, ch = image.shape
    mean = 0
    sigma = delta ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_jpeg_compression(image, quality):
    # Aplicar compressão JPEG
    pil_img = Image.fromarray(image)
    pil_img.save('temp.jpg', quality=quality)
    return cv2.imread('temp.jpg')

def generate_low_quality_image(high_quality_image):
    # Gerar imagem de baixa qualidade
    sigma = random.uniform(0.2, 10)
    r = random.randint(1, 8)
    delta = random.randint(0, 15)
    q = random.randint(60, 100)
    
    blurred = apply_gaussian_blur(high_quality_image, sigma)
    downsampled = downsample_image(blurred, r)
    noisy = add_gaussian_noise(downsampled, delta)
    low_quality_image = apply_jpeg_compression(noisy, q)
    
    return low_quality_image


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

