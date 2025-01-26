import numpy as np
import scipy.signal as signal

def main(image1:np.array, image2:np.array) -> np.array:
    """
    This function takes in two images and returns the difference between them.
    Using a convolution
    """
    y=min(image1.shape, image2.shape, key=lambda x:x[0])
    x=min(image1.shape, image2.shape, key=lambda x:x[1])
    image1=image1[(image1.shape[0]-y)//2:(image1.shape[0]-y)//2+y, (image1.shape[1]-x)//2:(image1.shape[1]-x)//2+x]
    image2=image2[(image2.shape[0]-y)//2:(image2.shape[0]-y)//2+y, (image2.shape[1]-x)//2:(image2.shape[1]-x)//2+x]
    signal.convolve2d(image1, image2, mode='same')
    convolved_difference = np.zeros_like(image1)

    # Apply convolution to each color channel separately
    kernel = np.ones((3, 3))  # Simple kernel for convolution
    for channel in range(3):  # Assuming RGB images
        difference = np.abs(image1[:, :, channel] - image2[:, :, channel])
        convolved_difference[:, :, channel] = signal.convolve2d(difference, kernel, mode='same', boundary='wrap')

    return convolved_difference