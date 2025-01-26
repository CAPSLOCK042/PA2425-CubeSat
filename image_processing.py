# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
import math

#def main(image_path: str , dimensions: tuple = (11, 8.5)) -> np.ndarray:
if True:
    """
    This function takes in an image path and dimensions of a grid and returns a processed image.
    """
    
    # def main(image_path="easySudoku.png"):
    image_path=r'C:\\Users\\russe\\CodeStuff\\paCubeSat\\PA2425-CubeSat\\images\\image_proccessing_184345.jpg'
    dimensions=(11,8.5)
    def find_quadrilateral(binary_image):
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
            cv2.drawContours(hull_image, [hull], -1, 1, thickness=-1)  #
            return hull_image
        print('fuck')
        # Fallback: No quadrilateral found
        chull=convex_hull_image(binary)
        if not chull.all():
            print('oof')
            chull=binary
        return chull
    
    def show_image(image, title="Image", cmap_type='gray'):
        plt.imshow(image, cmap=cmap_type)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def show_image_with_corners(image, coords, title="Corners detected"):
        coords = np.array(coords)
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
    def good_fix_coords(raw_coords, center, chull):
        
        def angle(p):
            if p[1]-center[1]>0:
                a=math.atan((-p[0] + center[0])/ (p[1] - center[1]))
                a= a if a>0 else a+2*np.pi
            else:
                a= math.atan((-p[0] + center[0])/ (p[1] - center[1]))
                a+=np.pi
            return a
        coords=sorted(raw_coords, key=lambda a: ((center[0]-a[0])**2+(center[0]-a[1])**2)**.5)[-4:]
        
        #order is in terms of dgrees from positive x axis
        coords=sorted(coords, key=angle)
        return coords
    
    def points6matrix(coords: np.array, dimensions: tuple, centroid, img_dim):
        # Note coords are in the form of (y,x)
        # y axis is also switched
    
        y = coords[1] - coords[2]
        x = coords[3] - coords[2]
        print('vectors')
        print(y, x)
    
        # Affine transformation works by matrix * m = b
        # m are the actual vectors of the rectangle
        m = np.zeros((2, 2))
        # b preferred vectors of the rectangle
        b = np.zeros((2, 2))
    
        b[0, 0] = dimensions[1]
        b[1, 1] = dimensions[0]
        m[:, 0] = y
        m[:, 1] = x
    
        # Matrix is the matrix needed to warp the image to the correct orientation
        matrix = np.matmul(b, np.linalg.inv(m))
        print("matrix", matrix)
    
        # Calculate the new centroid
        new_cent = np.matmul(matrix, centroid)
        ans = np.zeros((3, 3))
        ans[2, 2] = 1
        ans[:2, :2] = matrix
    
        # Calculate the translation needed to center the image
        translation = np.array(img_dim) // 2 - new_cent
        ans[0, 2] = translation[1]
        ans[1, 2] = translation[0]
        
        scale_factor = 10.5  # Adjust this factor as needed
        ans[:2, :2] *= scale_factor
        print(ans)
        return ans
            
    #updated one
    #make it so that is must take in dimesnsion as
    def points5matrix(coords:np.array, dimensions:tuple, centroid, img_dim): 
        #note coords are in the form of (y,x)
        #y axis is also switched
        
        y=coords[1]-coords[2]
        x=coords[3]-coords[2]
        print('vectors')
        print(y,x)

        #Affine tranfroamtion wokrs by matrix * m=b
        #m are the actual vectors of the rectangle
        m=np.zeros((2,2))
        #b preferred vectors of the rectangle
        b=np.zeros((2,2))
        
        b[0,0]=-dimensions[1]
        b[1,1]=dimensions[0]
        m[:,0]=y
        m[:,1]=x
        m[0]*=1
        #matrix is the matrix needed to warp the image to the correc torientation
        
        matrix=np.matmul(b, np.linalg.inv(m))
        print("matrix",matrix)
        x1=np.array(matrix)
        
        
        x1/=np.power(abs(np.linalg.det(x1)), .5)

        centroid[0]=centroid[0]
        new_cent=np.matmul(x1, centroid)
        ans=np.zeros((3,3))
        ans[2,2]=1
        ans[:2,:2]=x1
        a=new_cent-np.array(img_dim)//2
        print("asd")
        print(np.matmul(x1,y),np.matmul(x1,x))
        print(centroid, new_cent, a)
        ans[0,2]= img_dim[0]//2-new_cent[0]
        ans[1,2]= img_dim[1]//2-new_cent[1]
        print(ans)
        return ans
        
    

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
        
        
    image = 255 - cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_gray  = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    show_image(image , "og")
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8,8))
    cl1=clahe.apply(image_gray)
    image_gaussian = gaussian(image_gray)
    thresh = threshold_otsu(image_gaussian)
    binary = image_gaussian > thresh
    #binary=1-binary
    show_image(binary, "thresholded")

    """
    raw_coords = corner_peaks(corner_harris(binary), min_distance=2, threshold_rel=0.02)
    show_image_with_corners(binary, raw_coords)
    """
    
    chull = find_quadrilateral(binary)
    
    raw_coords = corner_peaks(corner_harris(
        chull), min_distance=20, threshold_rel=0.02)
    show_image(chull, "chull")
    #show_image_with_corners(chull, raw_coords)
    indices = np.argwhere(chull == 1)

    # Compute the centroid
    centroid = np.mean(indices, axis=0)
    coords = good_fix_coords(raw_coords, centroid, chull)
    show_image_with_corners(chull, coords)
    
    img_dim=chull.shape
    matrix=points5matrix(coords, dimensions, centroid, img_dim)
    tForm1 = AffineTransform(matrix)
    z=np.zeros((3,3))
    z[0,0]=1
    z[0,1]=-0.0530035
    z[1,1]=823322
    z[2,2]=1
    #tForm1 = AffineTransform(z)
    color_warp = warp(image, tForm1.inverse)
    image2 = warp(chull, tForm1.inverse)
    show_image(color_warp, "warped back?")
    input()
    # Check if image2 contains non-zero points before computing the convex hull
    thresh1 = threshold_otsu(image2)
    binary1= image2 > thresh1
    chull1 = find_quadrilateral(binary1)



    raw_coords1 = corner_peaks(corner_harris(
        chull1), min_distance=20, threshold_rel=0.02)
    show_image(chull1, "chull1")

    indices1 = np.argwhere(chull1 == 1)
    centroid1 = np.mean(indices1, axis=0)
    coords1 = good_fix_coords(raw_coords1, centroid1, chull1)
    show_image_with_corners(chull1, coords1)

    filtered_image = image2[int(coords1[1][0]):int(
        coords1[3][0]), int(coords1[1][1]):int(coords1[3][1])]
    color_ready_image=color_warp[int(coords1[1][0]):int(
        coords1[3][0]), int(coords1[1][1]):int(coords1[3][1])]
    show_image(color_ready_image, "ready image")
    # return ready_image

    #return color_ready_image
