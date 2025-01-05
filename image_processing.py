# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:46:23 2024

@author: russe
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 22:03:59 2024

@author: russe
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, gaussian
from skimage import data, color, measure
from skimage.feature import canny
from skimage.morphology import convex_hull_image
from skimage.feature import corner_harris, corner_peaks
from skimage.transform import warp, AffineTransform, resize

if True:
    # def main(image_path="easySudoku.png"):
    image_path = "easySudoku.png"
    image_path=r"C:\Users\russe\Pictures\Camera Roll\WIN_20250103_18_15_16_Pro.jpg"

    def show_image(image, title="Image", cmap_type='gray'):
        plt.imshow(image, cmap=cmap_type)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def show_image_with_corners(image, coords, title="Corners detected"):
        plt.imshow(image, interpolation='nearest', cmap='gray')
        plt.title(title)
        plt.plot(coords[:, 1], coords[:, 0], '+r', markersize=15)
        plt.axis('off')
        plt.show()
        
    #Needs to be fixed
    #should only return coords that are the maxes ie most left most right etc
    def fix_coords(raw_coords):
        raw_coords=np.array(raw_coords)
        l=min(raw_coords[:, 1])
        r=max(raw_coords[:, 1])
        u=min(raw_coords[:, 0])
        d=max(raw_coords[:, 0])
        
        #distance of a coordinate from point x y 
        s=lambda a,y,x: ((y-a[0])**2+(x-a[1])**2)**.5
        t=lambda a:(s(a,u,l), s(a,u,r), s(a,d,l), s(a,d,r))
        #Order should be most l-u, r-u, l-d
        a=np.array([t(raw_coords[i]) for i in range(len(raw_coords))])
        return [raw_coords[np.argmin(a[:,i])] for i in range (4)]
    
    def points5matrix(coords): 
        pass
    
    def points4matrix(coords):
        minx = np.where(coords[:, 1] == min(coords[:, 1]))[0][-1]
        if minx == np.where(coords[:, 0] == max(coords[:, 0]))[0][0]:
            ops = [0, 1, 2, 3]
            ops.remove(minx)
            # gets rid of max y value in case of a swuare will get rid of the top right most one
            ops.remove(np.where(coords[:, 0] == min(coords[:, 0]))[0][-1])
            p2 = minx
            p3 = ops[-1]
            p0 = ops[0]
            b = np.zeros((2, 2))
            b[0, 0] = -coords[p2][1]+coords[p3][1]
            b[1, 0] = coords[p2][0]-coords[p3][0]
            b[0, 1] = -coords[p2][1]+coords[p0][1]
            b[1, 1] = coords[p2][0]-coords[p0][0]
            a = np.zeros((3, 3))
            a[:2, :2] = b/np.power(np.linalg.det(b), .5)
            a[2, 2] = 1
            print(a)
            return a
        if coords[1][1] < coords[2][1]:
            p2 = 3
            p0 = 1
            p3 = 2
            b = np.zeros((2, 2))
            b[0, 0] = -coords[p2][1]+coords[p3][1]
            b[1, 0] = coords[p2][0]-coords[p3][0]
            b[0, 1] = -coords[p2][1]+coords[p0][1]
            b[1, 1] = coords[p2][0]-coords[p0][0]
            a = np.zeros((3, 3))
            a[:2, :2] = b/np.power(np.linalg.det(b), .5)
            a[2, 2] = 1
            print(a)
            return a
        else:
            p2 = 2
            p0 = 0
            p3 = 3
            b = np.zeros((2, 2))
            b[0, 0] = -coords[p2][1]+coords[p3][1]
            b[1, 0] = coords[p2][0]-coords[p3][0]
            b[0, 1] = -coords[p2][1]+coords[p0][1]
            b[1, 1] = coords[p2][0]-coords[p0][0]
            a = np.zeros((3, 3))
            a[:2, :2] = b/np.power(np.linalg.det(b), .5)
            a[2, 2] = 1
            print(a)
            return a
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_gray  = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    show_image(image_gray , "og")
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8,8))
    cl1=clahe.apply(image_gray)
    image_gaussian = gaussian(image_gray)
    thresh = threshold_otsu(image_gaussian)
    binary = image_gaussian > thresh
    show_image(binary, "thresholded")
    input()

    """
    raw_coords = corner_peaks(corner_harris(binary), min_distance=2, threshold_rel=0.02)
    show_image_with_corners(binary, raw_coords)
    """

    chull = convex_hull_image(binary)
    show_image(chull, "shape")

    raw_coords = corner_peaks(corner_harris(
        chull), min_distance=2, threshold_rel=0.02)
    # show_image(chull)
    show_image_with_corners(chull, raw_coords)

    coords = np.array(fix_coords(raw_coords))


    tForm1 = AffineTransform(points4matrix(coords))
    image2 = warp(binary, tForm1.inverse)
    show_image(image2, "warped back?")

    chull1 = convex_hull_image(image2)
    raw_coords1 = corner_peaks(corner_harris(
        chull1), min_distance=2, threshold_rel=0.02)
    coords1 = fix_coords(raw_coords1)
    image2 = warp(binary, tForm1.inverse)
    filtered_image = image2[int(coords1[0][0]):int(
        coords1[2][0]), int(coords1[0][1]):int(coords1[1][1])]
    b = len(filtered_image)+9-len(filtered_image) % 9
    ready_image = resize(filtered_image, (b, b))
    show_image(ready_image)
    color_ready_image = color.gray2rgb(ready_image).astype(np.float64)
    show_image(color_ready_image, "ready image")
    # return ready_image

    # return color_ready_image
# main()
