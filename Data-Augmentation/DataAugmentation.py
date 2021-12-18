import os
import pathlib as pl
from PIL import Image 

class DataAugmentation:
    """
    Handles all data augmentation jobs
    """

    defaultImageProcessingOptions = {
        'position': {
            'translation': {
                'active': True,
                'iterations': 50,
            },
            'rotation': {
                'active': True,
                'iterations': 90
            },
            'cropping': {
                'active': True,
                'iterations': 50
            },
            'flipping': {
                'active': True
            },
            'scaling': {
                'active': True, 
                'iterations': 50,
                'min': 0.25,
                'max': 3,
            },
            'shearing': {
                'active': True, 
                'iterations': 50,
                'maxAngle': 3,
            }             
        },
        'color': {
            'brightness': {
                'active': True,
                'iterations': 50,
                'min': 0.2,
                'max': 2
            },
            'contrast': {
                'active': True,
                'iterations': 50
            },
            'saturation': {
                'active': True,
                'iterations': 50
            },
            'hue': {
                'active': True,
                'iterations': 50
            }
        },
        'noise': {
            'gaussian': {
                'active': True,
                'iterations': 50
            },
            'saltpepper': {
                'active': True,
                'iterations': 50
            },
            'speckle': {
                'active': True,
                'iterations': 50
            }
        },
        'grayscale': True,
        'repeatForGrayscale': True
    }

    def __init__(self, rescale_size=256) -> None: 
        self.rescale_size = rescale_size
        pass

    def analyzeDirectory(self, path, only_images=True):
        """
        Analyzes the contents of a directory and outlines the most important information for each file.
        Returns an array with information for each directory. 
        Each directory information contains another array with a list of each file.
        The information for each file is stored in a dictionary with the following structure:
            - filename
            - type (jpeg, png, etc...)
            - dimensions (width/height tuple)
            - file size (in kB)
        """

        info = []
        for entry in os.scandir(path):
            if entry.is_file:
                if entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    info.append({
                        'filename': entry.name,
                        'type': pl.Path(entry).suffix[1:],
                        'dimensions': Image.open(os.path.abspath(path + '\\' + entry.name)).size,
                        'size': entry.stat().st_size / 1024
                    })

        return info

        pass

    def processImageBatch(self, batch, options=defaultImageProcessingOptions):
        """
        Processes a batch of images 
        Returns: Processed images as a list
        """
        pass

    def processImage(self, image, options=defaultImageProcessingOptions):
        """
        Processes a single image 
        Returns: Processed image
        """
        pass

    def makeBatches(self, directory_info, batch_size=100):
        """
        Uses the information given by the analyzeDirectory function to make batches of images to load
        Returns: Batches of images as their file location (Doesn't load the image data into memory since that would defeat its purpose!)
        """
        pass

    def process(self, path='..\\images', options={'imageProcessing': defaultImageProcessingOptions}):
        """
        Processes all images inside the given directory. 
        Function expects to find different images of each class to classify in its own subdirectory and will generate new subdirectories for the new images
        """
        pass
    pass