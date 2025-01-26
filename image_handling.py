

from git import Repo
from image_processing import main
import os
import matplotlib.pyplot as plt

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
    repo = Repo(REPO_PATH)
    origin = repo.remote('origin')

    image_folder_path = os.path.join(REPO_PATH, FOLDER_PATH)
    image_paths = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if f.endswith('.jpg')]

    return image_paths
    
a=returnImages()
b=[c for c in a if "image_" in c]
c=main(b[-3], (11,8.5))

show_image(c)
