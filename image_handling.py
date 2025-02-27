

from git import Repo
from image_processing import main
import os
import matplotlib.pyplot as plt
from skimage.color import gray2rgb
import image_difference as img_diff
import numpy as np
import cv2

REPO_PATH = r"\Users\russe\CodeStuff\paCubeSat\PA2425-CubeSat"     #Your github repo path: ex. /home/pi/FlatSatChallenge
FOLDER_PATH = "images" 
def show_image(image, title="Image", cmap_type='gray'):
        plt.imshow(image, cmap=cmap_type)
        plt.title(title)
        plt.axis('off')
        plt.show()
def returnImages() -> list[str]:
    """
    This function returns a list of image paths.
    """
    #repo = Repo(REPO_PATH)
    #origin = repo.remote('origin')

    image_folder_path = os.path.join(REPO_PATH, FOLDER_PATH)
    image_paths = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if f.endswith('.jpg')]

    return image_paths
    
a=returnImages()

image_path1=r"C:\Users\russe\Downloads\spain_before.webp"
image_path2=r"C:\Users\russe\Downloads\spain_after.webp"

color1 = cv2.cvtColor(cv2.imread(image_path1), cv2.COLOR_BGR2RGB)
color2 = cv2.cvtColor(cv2.imread(image_path2), cv2.COLOR_BGR2RGB)
#color1=main(image_path=r"C:\Users\russe\Pictures\Screenshots\test1.png")
#color2=main(image_path=r"C:\Users\russe\Pictures\Screenshots\test2.png")
image_path1=REPO_PATH+"/images/buttontest_182819.jpg"
image_path2=REPO_PATH+"/images/buttontest_183103.jpg"

color1=main(image_path1)
color2=main(image_path2)

show_image(color1)
show_image(color2)
a=img_diff.main(color1, color2)
show_image(a)
"""
images/buttontest_182819.jpg
images/buttontest_183103.jpg
"""