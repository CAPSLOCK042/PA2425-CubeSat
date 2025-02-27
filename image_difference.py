import numpy as np
import scipy.signal as signal
from skimage.transform import resize
from skimage.filters import threshold_otsu, gaussian
import matplotlib.pyplot as plt
import cv2



def main(image1: np.array, image2: np.array) -> np.array:
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
    #to ignore clouds
    cloud_threshold=1
    mask1 = np.std(image1, axis=2) > cloud_threshold
    mask2 = np.std(image2, axis=2) > cloud_threshold

    # Combine masks
    combined_mask = np.logical_or(mask1, mask2)
    print(combined_mask.shape)
    # Initialize arrays to store the convolved differences
    convolved_difference1 = np.zeros((image1.shape[0], image1.shape[1]))

    # Define a Gaussian blur kernel
    kernel = np.array([[1,  4,  6,  4, 1],
                       [4, 16, 24, 16, 4],
                       [6, 24, 36, 24, 6],
                       [4, 16, 24, 16, 4],
                       [1,  4,  6,  4, 1]]) / 256  # Gaussian blur kernel
    
    # Define the strength of the convolution for each color channel
    strength = [5, 4, 1]
    
    # Apply convolution to each color channel separately
    for channel in range(3):  # Assuming RGB images
        # Calculate the absolute difference between the two images for the current channel
        difference = gaussian(np.abs(np.squeeze(image1[:, :, channel] - image2[:, :, channel])))
        difference[combined_mask] = 0
        # Accumulate the weighted convolved differences for each channel
        convolved_difference1[:, :] += strength[channel] * signal.convolve2d(difference, kernel, mode='same', boundary='wrap')
    print(convolved_difference1.shape)
    return convolved_difference1

