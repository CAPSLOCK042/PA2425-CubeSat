# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, gaussian
from skimage.feature import corner_peaks, corner_harris
from skimage.morphology import convex_hull_image
from skimage.transform import rescale
import math


def find_quadrilateral4(binary_image):
    """
    Finds the largest quadrilateral in a binary image.
    
    Args:
        binary_image (np.array): The input binary image.
    
    Returns:
        np.array: The image with the largest quadrilateral drawn around the largest contour.
        list: The coordinates of the quadrilateral corners.
        float: The area of the quadrilateral.
    """
    # Convert binary image to uint8 for OpenCV compatibility
    binary_uint8 = binary_image.astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Initialize variable to store the largest quadrilateral
    largest_quad = None
    largest_area = 0
    
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if the approximated contour has 4 vertices (is a quadrilateral)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area< (binary_uint8.shape[0]*binary_uint8.shape[1]*.95*.95) and area > largest_area:
                largest_area = area
                largest_quad = approx
    
    # Draw the largest quadrilateral
    drawing = np.zeros((binary_uint8.shape[0], binary_uint8.shape[1], 3), dtype=np.uint8)
    if largest_quad is not None:
        color = (255, 255, 255)
        cv2.drawContours(drawing, [largest_quad], 0, color, 2)
    
    return drawing, [np.flip(i) for i in np.squeeze(largest_quad)], largest_area

def find_quadrilateral3(binary_image):
    """
    Finds the rotated rectangles and ellipses for each contour in a binary image.
    
    Args:
        binary_image (np.array): The input binary image.
    
    Returns:
        np.array: The image with the largest rotated rectangle drawn around the contours.
        np.array: The largest rotated rectangle coordinates 
        float: The largest rotated rectangle area
    """
    # Convert binary image to uint8 for OpenCV compatibility
    binary_uint8 = binary_image.astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours + rotated rects + ellipses
    drawing = np.zeros((binary_uint8.shape[0], binary_uint8.shape[1], 3), dtype=np.uint8)
    def area(contour):
        box = cv2.boxPoints(cv2.minAreaRect(contour))
        coords = np.intp(box)
        area = 0
        for i in range(len(coords)):
            a = np.zeros((2, 2))
            a[:, 0] = coords[i - 1]
            a[:, 1] = coords[i]
            area += np.linalg.det(a)
        return round(area/2,2)
    aa=[(np.intp(cv2.boxPoints(cv2.minAreaRect(i))),area(i)) for i in contours ]
    box=max(aa,key=lambda x:x[1] if x[1]<binary_uint8.shape[0]*binary_uint8.shape[1]*.95*.95 else 0)
    cv2.drawContours(drawing, [box[0]], 0, (255,255,255))
    print(box)
    return drawing,[np.flip(i) for i in np.squeeze(box[0])], box[1]



def show_image(image, title="Image", cmap_type='gray'):
    """
    Displays an image using matplotlib.
    
    Args:
        image (np.array): The input image.
        title (str): The title of the image.
        cmap_type (str): The colormap type.
    """
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_image_with_corners(image, coords, title="Corners detected"):
    """
    Displays an image with detected corners using matplotlib.
    
    Args:
        image (np.array): The input image.
        coords (list): The coordinates of the corners.
        title (str): The title of the image.
    """
    coords = np.array(coords)
    plt.imshow(image, interpolation='nearest', cmap='gray')
    plt.title(title)
    plt.plot(coords[:, 1], coords[:, 0], '+r', markersize=15)
    plt.axis('off')
    plt.show()

def good_fix_coords(raw_coords, center, chull):
    """
    Fixes the coordinates of the corners based on their distance from the center.
    
    Args:
        raw_coords (list): The raw coordinates of the corners.
        center (tuple): The center of the convex hull.
        chull (np.array): The convex hull image.
    
    Returns:
        list: The fixed coordinates of the corners.
    """
    def angle(p):
        if p[1] - center[1] > 0:
            a = math.atan((-p[0] + center[0]) / (p[1] - center[1]))
        else:
            a = math.atan((-p[0] + center[0]) / (p[1] - center[1]))
            a += np.pi
        a -= 1 * np.pi / 2
        a = a if a > 0 else a + 2 * np.pi
        return a
    
    coords = sorted(raw_coords, key=lambda a: ((center[0] - a[0]) ** 2 + (center[1] - a[1]) ** 2) ** .5)[-4:]
    
    # Order is in terms of degrees from positive x axis
    coords = sorted(coords, key=angle)
    return coords

def points8matrix(coords, area, dimensions):
    """
    Computes the affine transformation matrix from the given coordinates and dimensions.
    
    Args:
        coords (list): The coordinates of the corners.
        area (float): The area of the convex hull.
        dimensions (tuple): The dimensions of the output image.
    
    Returns:
        np.array: The affine transformation matrix.
    """
    
    x = (area / (dimensions[0] * dimensions[1])) ** .5
    height = round(dimensions[0] * x)
    width = round(dimensions[1] * x)
    rect_coords = np.array([
        [0, 0],
        [height, 0],
        [height, width]
    ])
    
    # Construct the system of equations
    A = []
    b = []
    for (x, y), (X, Y) in zip(coords[:3], rect_coords):
        A.append([x, y, 0, 0, 1, 0])
        A.append([0, 0, x, y, 0, 1])
        b.append(X)
        b.append(Y)
    A = np.array(A)
    affine_params = np.linalg.solve(A, b)

    # Extract the affine transformation matrix
    a, b, c, d, e, f = affine_params

    affine_matrix = np.array([
        [a, b, e],
        [c, d, f],
        [0, 0, 1]
    ])
    return affine_matrix



def points7matrix(coords, dimensions):
    """
    Computes the affine transformation matrix from the given coordinates and dimensions.
    
    Args:
        coords (list): The coordinates of the corners.
        dimensions (tuple): The dimensions of the output image.
    
    Returns:
        np.array: The affine transformation matrix.
    """
    area = 0
    for i in range(len(coords)):
        a = np.zeros((2, 2))
        a[:, 0] = coords[i - 1]
        a[:, 1] = coords[i]
        area += np.linalg.det(a)
    area /= 2
    x = (area / (dimensions[0] * dimensions[1])) ** .5
    height = round(dimensions[0] * x)
    width = round(dimensions[1] * x)
    rect_coords = np.array([
        [0, 0],
        [height, 0],
        [height, width]
    ])
    
    # Construct the system of equations
    A = []
    b = []
    for (x, y), (X, Y) in zip(coords[:3], rect_coords):
        A.append([x, y, 0, 0, 1, 0])
        A.append([0, 0, x, y, 0, 1])
        b.append(X)
        b.append(Y)
    A = np.array(A)
    affine_params = np.linalg.solve(A, b)

    # Extract the affine transformation matrix
    a, b, c, d, e, f = affine_params

    affine_matrix = np.array([
        [a, b, e],
        [c, d, f],
        [0, 0, 1]
    ])
    return affine_matrix

def apply_affine_transform(point, matrix):
    """
    Applies an affine transformation to a single point.
    
    Args:
        point (tuple): The (y, x) coordinates of the point.
        matrix (np.array): The 3x3 affine transformation matrix.
    
    Returns:
        np.array: The transformed (y, x) coordinates.
    """
    y, x = point
    transformed = matrix @ np.array([y, x, 1])  # Apply the transformation
    return transformed[:2]  # Return only the (x, y) coordinates

def gaussian_kernel(n=3, mean_x=1.0, mean_y=1.0, sigma=.3):
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

def colour(image, point, n=3):
    """
    Computes the color of a point in the image using a Gaussian kernel.

    Args:
        image (np.array): The input image.
        point (tuple): The (y, x) coordinates of the point.
        n (int): The kernel size.

    Returns:
        float: The color value of the point.
    """
    mean_y = point[0] - round(point[0])
    mean_x = point[1] - round(point[1])
    conv = gaussian_kernel(n=n, mean_x=1 + mean_x, mean_y=1 - mean_y)
    coloury = [conv[i, j] * image[round(point[0]) - 1 + i, round(point[1]) - 1 + j] for i in range(n) for j in range(n)]
    c = sum(coloury)
    
    return c

def affine_warp2(image, matrix, coords, isColor, sigma=1):
    """
    Applies an affine warp to the image using the given transformation matrix.

    Args:
        image (np.array): The input image.
        matrix (np.array): The affine transformation matrix.
        coords (list): The coordinates of the corners.
        isColor (bool): Whether the image is in color.
        sigma (float): The scaling factor.

    Returns:
        np.array: The warped image.
    """
    height, width = image.shape[:2]
    if len(image.shape) == 2:
        image = np.reshape(image, (height, width, 1))
    height1, width1 = apply_affine_transform(coords[2], matrix)
    height1 = round(height1)
    width1 = round(width1)
    warped_image = np.zeros((int((height1 + 1) * sigma), int((width1 + 1) * sigma), image.shape[2]), dtype=np.uint8 if isColor else np.float64)
    inv = np.linalg.inv(matrix)
    for y in range(int(height1 + 1) * sigma):
        for x in range(int((width1 + 1) * sigma)):
            y1, x1 = apply_affine_transform((y / sigma, x / sigma), inv)
            color = [colour(image[:, :, i], (y1, x1)) for i in range(image.shape[2])]
            warped_image[y, x] = np.array(color, dtype=np.uint8 if isColor else np.float64)
    return warped_image

def main(image_path=r'C:\\Users\\russe\\CodeStuff\\paCubeSat\\PA2425-CubeSat\\images\\image_proccessing_184345.jpg', dimensions=(8.5, 11), sigma=1):

    """
    Main function to process the image and apply affine transformation.

    Args:
        image_path (str): The path to the input image.
        dimensions (tuple): The dimensions of the output image.
        sigma (float): The scaling factor.

    Returns:
        tuple: The processed grayscale and color images.
    """
    # Load and preprocess the image
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    show_image(image, "Original Image")
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
    image_gaussian = gaussian(image_gray)
    thresh = threshold_otsu(image_gaussian)
    binary = image_gaussian > thresh
    show_image(binary, "Thresholded Image")
    """
    # Find the convex hull of the binary image
    chull,raw_coords,area  = find_quadrilateral3(binary)
    show_image(chull, "Convex Hull")
    
    # Find the centroid and fix the coordinates of the corners
    centroid = np.mean(raw_coords, axis=0)
    coords = good_fix_coords(raw_coords, centroid, chull)
    show_image_with_corners(chull, coords, "Corners Detected")
    
    # Compute the affine transformation matrix
    matrix = points8matrix(coords,area, dimensions)
    """
    chull,raw_coords,area  = find_quadrilateral4(binary)
    show_image(chull,"chull")
    

    # Compute the centroid
    centroid = np.mean(raw_coords, axis=0)
    coords = good_fix_coords(raw_coords, centroid, chull)
    show_image_with_corners(chull, coords)
    
    matrix=points8matrix(coords, area,dimensions)
    # Apply the affine warp to the grayscale and color images

    color_image = affine_warp2(image, matrix, coords, True)
    show_image(color_image, "Please fucking work color")
    return color_image