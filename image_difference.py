import numpy as np
import scipy.signal as signal
from skimage.transform import resize
from skimage.filters import threshold_otsu, gaussian
import matplotlib.pyplot as plt
import cv2

def gaussian_kernel(n=7, mean_x=1.0, mean_y=1.0, sigma=.6):
    """
    Create a 3x3 Gaussian kernel with a slightly off-center peak.

    Args:
        n (int): The kernel size.
        mean_x (float): The x-coordinate of the Gaussian mean (default: 1.0).
        mean_y (float): The y-coordinate of the Gaussian mean (default: 1.0).
        sigma (float): The standard deviation of the Gaussian (default: 1.0).

    Returns:
        np.array: A 3x3 Gaussian kernel.
    """
    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    # Compute the Gaussian function
    kernel = np.exp(-((x - mean_x)**2 + (y - mean_y)**2) / (2 * sigma**2))
    # Normalize the kernel so that the sum is 1
    kernel /= np.sum(kernel)
    return kernel

def main(image1: np.array, image2: np.array, strength) -> np.array:
    """
    This function takes in two images and returns the difference between them
    using a convolution with a Gaussian blur kernel, focusing on the blue channel
    and excluding areas with high intensity in all channels (white clouds).

    Args:
        image1 (np.array): The first input image.
        image2 (np.array): The second input image.

    Returns:
        np.array: The convolved difference between the two images.
    """
    # Rescale images to the mean of their array shape
    mean_shape = np.mean([image1.shape, image2.shape], axis=0).astype(int)
    print(mean_shape)
    image1 = resize(image1, (mean_shape[0], mean_shape[1],), anti_aliasing=True)
    image2 = resize(image2, (mean_shape[0], mean_shape[1]), anti_aliasing=True)
    
    # Initialize arrays to store the convolved differences
    convolved_difference1 = np.zeros((image1.shape[0], image1.shape[1]))

    # Define a Gaussian blur kernel
    kernel =gaussian_kernel(n=19, sigma=1)  # Gaussian blur kernel
    
    # Define the strength of the convolution for each color channel
    
    
    
    
    # Apply convolution to each color channel separately
    for channel in range(3):  # Assuming RGB images
    #gaussian
        # Calculate the absolute difference between the two images for the current channel
        difference = (np.abs(np.squeeze(image1[:, :, channel] - image2[:, :, channel])))
        #difference[combined_mask] = 0
        # Accumulate the weighted convolved differences for each channel
        convolved_difference1[:, :] += strength[channel] * signal.convolve2d(difference, kernel, mode='same', boundary='wrap')
    
    print(convolved_difference1.shape)
    
    #return total
    # Normalize the convolved difference to the range [0, 1]
    convolved_difference1 = (convolved_difference1 - np.min(convolved_difference1)) / (np.max(convolved_difference1) - np.min(convolved_difference1))
    
    # Increase brightness by scaling the normalized difference
    brightness_factor = 1.5  # Adjust this factor to increase brightness
    brightened_image = np.clip(convolved_difference1 * brightness_factor, 0, 1)
    
    # Convert to higher dynamic range
    brightened_image = (brightened_image * 255).astype(np.uint8)
    
    return brightened_image

