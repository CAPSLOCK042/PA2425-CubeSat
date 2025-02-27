# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:23:02 2025

@author: russe
"""

import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, gaussian
from skimage.feature import corner_peaks, corner_harris
from skimage.morphology import convex_hull_image
from skimage.transform import rescale
import math
import random as rng

rng.seed(12345)

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
    contours, _ = cv.findContours(binary_uint8, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    
    # Initialize variable to store the largest quadrilateral
    largest_quad = None
    largest_area = 0
    
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        
        # Check if the approximated contour has 4 vertices (is a quadrilateral)
        if len(approx) == 4:
            area = cv.contourArea(approx)
            if area< (binary.shape[0]*binary.shape[1]*.95*.95) and area > largest_area:
                largest_area = area
                largest_quad = approx
    
    # Draw the largest quadrilateral
    drawing = np.zeros((binary_uint8.shape[0], binary_uint8.shape[1], 3), dtype=np.uint8)
    if largest_quad is not None:
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv.drawContours(drawing, [largest_quad], 0, color, 2)
    
    return drawing, [np.flip(i) for i in np.squeeze(largest_quad)], largest_area

def find_quadrilateral1(binary_image):
    """
    Finds the best option for a parallelogram around the largest contour in a binary image.
    
    Args:
        binary_image (np.array): The input binary image.
    
    Returns:
        np.array: The image with the parallelogram drawn around the largest contour.
    """
    # Convert binary image to uint8 for OpenCV compatibility
    binary_uint8 = binary_image.astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Initialize variables to store the best bounding rectangle
    best_rect = None
    best_area = 0
    
    # Iterate through contours to find the best bounding rectangle
    for contour in contours:
        # Approximate the contour to a polygon
        contour_poly = cv2.approxPolyDP(contour, 3, True)
        # Get the bounding rectangle for the polygon
        bound_rect = cv2.boundingRect(contour_poly)
        # Calculate the area of the bounding rectangle
        rect_area = bound_rect[2] * bound_rect[3]
        # Update the best bounding rectangle if the current one is larger
        if rect_area > best_area:
            best_area = rect_area
            best_rect = bound_rect
    
    # Create an empty image to draw the parallelogram
    drawing = np.zeros((binary_uint8.shape[0], binary_uint8.shape[1], 3), dtype=np.uint8)
    
    # Draw the best bounding rectangle as a parallelogram
    if best_rect is not None:
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv2.rectangle(drawing, (int(best_rect[0]), int(best_rect[1])),
                      (int(best_rect[0] + best_rect[2]), int(best_rect[1] + best_rect[3])), color, 2)
    
    return drawing

def find_quadrilateral(binary_image):
    """
    Finds the convex hull of the largest contour in a binary image.
    
    Args:
        binary_image (np.array): The input binary image.
    
    Returns:
        np.array: The convex hull image.
    """
    # Convert binary image to uint8 for OpenCV compatibility
    binary_uint8 = binary_image.astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    hull_image = np.zeros_like(binary_uint8)
    
    # Compute convex hull for the largest contour
    if contours:
        hull = cv2.convexHull(contours[0])
        cv2.drawContours(hull_image, [hull], -1, 1, thickness=-1)
        return hull_image
    
    # Fallback: No quadrilateral found
    chull = convex_hull_image(binary_image)
    if not chull.all():
        chull = binary_image
    return chull
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
    contours, _ = cv.findContours(binary_uint8, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Draw contours + rotated rects + ellipses
    drawing = np.zeros((binary_uint8.shape[0], binary_uint8.shape[1], 3), dtype=np.uint8)
    def area(contour):
        box = cv.boxPoints(cv.minAreaRect(contour))
        coords = np.intp(box)
        area = 0
        for i in range(len(coords)):
            a = np.zeros((2, 2))
            a[:, 0] = coords[i - 1]
            a[:, 1] = coords[i]
            area += np.linalg.det(a)
        return round(area/2,2)
    aa=[(np.intp(cv.boxPoints(cv.minAreaRect(i))),area(i)) for i in contours ]
    box=max(aa,key=lambda x:x[1] if x[1]<binary_uint8.shape[0]*binary_uint8.shape[1]*.95*.95 else 0)
    cv.drawContours(drawing, [box[0]], 0, (255,255,255))
    print(box)
    return drawing,box



def find_quadrilateral2(binary_image):
    """
    Finds the largest rotated rectangle for the largest contour in a binary image.
    
    Args:
        binary_image (np.array): The input binary image.
    
    Returns:
        np.array: The image with the largest rotated rectangle drawn around the largest contour.
    """
    # Convert binary image to uint8 for OpenCV compatibility
    binary_uint8 = binary_image.astype(np.uint8)
    
    # Find contours
    contours, _ = cv.findContours(binary_uint8, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    
    # Initialize variable to store the largest rotated rectangle
    largest_rect = None
    
    if contours:
        # Find the largest rotated rectangle for the largest contour
        largest_rect = cv.minAreaRect(contours[0])
    
    # Draw the largest rotated rectangle
    drawing = np.zeros((binary_uint8.shape[0], binary_uint8.shape[1], 3), dtype=np.uint8)
    if largest_rect is not None:
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        box = cv.boxPoints(largest_rect)
        box = np.intp(box)
        cv.drawContours(drawing, [box], 0, color)
    
    return drawing

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


def best_find_quad(binary):
    a=find_quadrilateral4(binary)
    

image_path=r"C:\Users\russe\Pictures\Screenshots\test2.png"
#image_path=r'C:\\Users\\russe\\CodeStuff\\paCubeSat\\PA2425-CubeSat\\images\\image_proccessing_184345.jpg'

image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
show_image(image, "Original Image")


image_gaussian = gaussian(image_gray)
thresh = threshold_otsu(image_gaussian)
binary = image_gaussian > thresh
# Show results of different find_quadrilateral functions
a = find_quadrilateral(binary)
show_image(a, "find_quadrilateral")

b = find_quadrilateral1(binary)
show_image(b, "find_quadrilateral1")

c = find_quadrilateral2(binary)
show_image(c, "find_quadrilateral2")

d,aa = find_quadrilateral3(binary)
show_image(d, "find_quadrilateral3")

e,ab,abb = find_quadrilateral4(binary)
show_image(e, "find_quadrilateral4")