import numpy as np
import cv2


def add_noise(image, noise_type='gaussian', scale=0.1, sigma=30, salt_pepper_amount=0.05):
    """
    Add non-Gaussian and Gaussian noise to an image.

    Args:
    - image: Input image (numpy array).
    - noise_type: Type of non-Gaussian noise. Options: 'gaussian', 'uniform', 'exponential', 'poisson', 'salt_pepper'.
    - scale: Scaling factor to adjust the magnitude of the noise.
    - sigma: Standard deviation of Gaussian noise.
    - salt_pepper_amount: Probability of adding salt or pepper noise.

    Returns:
    - Image with added non-Gaussian noise.
    """
    if noise_type == 'uniform':
        noise = np.random.uniform(low=-scale, high=scale, size=image.shape)
    elif noise_type == 'exponential':
        noise = np.random.exponential(scale, size=image.shape)
    elif noise_type == 'poisson':
        noise = np.random.poisson(scale, size=image.shape)
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
