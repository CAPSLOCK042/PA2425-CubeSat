# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 18:57:57 2025

@author: russe
"""

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
        cv2.drawContours(hull_image, [hull], -1, 1, thickness=-1)
        return hull_image
    
    # Fallback: No quadrilateral found
    chull = convex_hull_image(binary_image)
    if not chull.all():
        chull = binary_image
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

def good_fix_coords(raw_coords, center, chull):
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

def points8matrix(coords, dimensions):
    area = 0
    for i in range(len(coords)):
        a = np.zeros((2, 2))
        a[:, 0] = coords[i - 1]
        a[:, 1] = coords[i]
        area += np.linalg.det(a)
    area /= 2
    
    line30=coords[3]-coords[0]
    line01=coords[0]-coords[1]
    line12=coords[1]-coords[2]
    
    


#works alright finds the correct equations 
def points7matrix(coords, dimensions):
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
    transformed = matrix @ np.array([y,x,1])  # Apply the transformation
    return transformed[:2]  # Return only the (x, y) coordinates
def gaussian_kernel(n=3, mean_x=1.0, mean_y=1.0, sigma=.2):
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
    mean_y=point[0]-round(point[0])
    mean_x=point[1]-round(point[1])
    conv=gaussian_kernel(n=n, mean_x=mean_x, mean_y=-mean_y)
    coloury=[conv[point[0]-1+i,point[1]-1+j]*image[i,j] for i in range(n) for j in range(n)]
    c=sum(coloury)

    return c
def affine_warp3(image, matrix,coords):
    pass

#works alright
def affine_warp2(image, matrix, coords):
    height, width = image.shape[:2]
    if len(image.shape)==2:
        image=np.reshape(image, (height, width, 1))
    height1, width1 = apply_affine_transform(coords[2], matrix)
    height1=round(height1)
    width1=round(width1)
    warped_image=np.zeros((height1+1, width1+1, image.shape[2]))
    for y in range(height1+1):
        for x in range(width1+1):
            
            y1,x1=apply_affine_transform((y,x), np.linalg.inv(matrix))
            warped_image[y,x]=image[int(y1),int(x1)]
            continue
            color=[colour(image[:,:,i], (y1,x1)) for i in range(image.shape[2])]
            warped_image[y,x]=color if len(color)>1 else color[0]
    return warped_image
#dont use
def affine_warp1(image, matrix, coords, output_shape):
    height, width = image.shape[:2]
    print(coords, [apply_affine_transform(i, matrix) for i in coords])
    newHeight, newWidth = apply_affine_transform(coords[2], matrix)

    # Initialize the warped image with the same number of channels as the input image
    warped_image = np.zeros((int(newHeight)*100, int(newWidth)*100, chull.shape[2] if len(chull.shape) >= 3 else 1), dtype=object)
    for y in range(height):
        for x in range(width):
            y1, x1 = apply_affine_transform((y, x), matrix)
            y1*=100
            x1*=100
            y1=int(y1)
            x1=int(x1)
            if y1>=0 and x1>=0 and y1<newHeight and x1 <newWidth:
                try:
                    warped_image[y1, x1] = image[y,x]
                except:
                    print("error", y1,x1)
                    input()
    return rescale(warped_image, .01, anti_aliasing=True)
            
#don't use
def affine_warp(image, matrix, output_shape):
    """
    Warps an image using an affine transformation matrix.
    
    Args:
        image (np.array): The input image to warp.
        matrix (np.array): The 3x3 affine transformation matrix.
        output_shape (tuple): The (height, width) of the output image.
    
    Returns:
        np.array: The warped image.
    """
    height, width = output_shape
    warped_image = np.zeros((height, width, image.shape[2]), dtype=image.dtype)  # Initialize output image
    
    # Create a grid of coordinates for the output image
    y_coords, x_coords = np.indices((height, width))
    coords = np.stack([x_coords.flatten(), y_coords.flatten()], axis=-1)  # Shape: (N, 2)
    
    # Apply the inverse affine transformation to map output coordinates to input coordinates
    inverse_matrix = np.linalg.inv(matrix)  # Compute the inverse of the affine matrix
    transformed_coords = np.array([apply_affine_transform(p, inverse_matrix) for p in coords])
    
    # Reshape the transformed coordinates to match the output image shape
    transformed_coords = transformed_coords.reshape((height, width, 2))
    
    # Perform bilinear interpolation to sample the input image
    for y in range(height):
        for x in range(width):
            src_x, src_y = transformed_coords[y, x]
            
            # Check if the transformed coordinates are within the input image bounds
            if 0 <= src_x < image.shape[1] and 0 <= src_y < image.shape[0]:
                warped_image[y, x] = cv2.getRectSubPix(image, (1, 1), (src_x, src_y))
    
    return warped_image

def main(image_path, dimensions):
    image = 255 - cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    show_image(image, "Original Image")
    
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(image_gray)
    image_gaussian = gaussian(image_gray)
    thresh = threshold_otsu(image_gaussian)
    binary = image_gaussian > thresh
    show_image(binary, "Thresholded Image")

    chull = find_quadrilateral(binary)
    raw_coords = corner_peaks(corner_harris(chull), min_distance=20, threshold_rel=0.02)
    show_image(chull, "Convex Hull")
    
    indices = np.argwhere(chull == 1)
    centroid = np.mean(indices, axis=0)
    coords = good_fix_coords(raw_coords, centroid, chull)
    show_image_with_corners(chull, coords, "Corners Detected")
    
    img_dim = chull.shape
    matrix = points7matrix(coords, dimensions)
    
    color_warp = cv2.warpAffine(image, matrix[:2, :], img_dim)
    image2 = cv2.warpAffine(chull, matrix[:2, :], img_dim)
    show_image(color_warp, "Warped Image")
    show_image(affine_warp1(image, matrix, img_dim), "Please fucking work")

if __name__ == "__main__":
    image_path = r'C:\\Users\\russe\\CodeStuff\\paCubeSat\\PA2425-CubeSat\\images\\image_proccessing_184345.jpg'
    dimensions = (11, 8.5)
    #main(image_path, dimensions)
    image = 255 - cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    show_image(image, "Original Image")
    
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(image_gray)
    image_gaussian = gaussian(image_gray)
    thresh = threshold_otsu(image_gaussian)
    binary = image_gaussian > thresh
    show_image(binary, "Thresholded Image")

    chull = find_quadrilateral(binary)
    raw_coords = corner_peaks(corner_harris(chull), min_distance=20, threshold_rel=0.02)
    show_image(chull, "Convex Hull")
    
    indices = np.argwhere(chull == 1)
    centroid = np.mean(indices, axis=0)
    coords = good_fix_coords(raw_coords, centroid, chull)
    show_image_with_corners(chull, coords, "Corners Detected")
    
    img_dim = chull.shape
    matrix = points7matrix(coords, dimensions)
    
    color_warp = cv2.warpAffine(image, matrix[:2, :], img_dim)
    image2 = cv2.warpAffine(chull, matrix[:2, :], img_dim)
    show_image(color_warp, "Warped Image")
    imageish=affine_warp2(chull, matrix, coords)
    show_image(imageish, "Please fucking work")



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
