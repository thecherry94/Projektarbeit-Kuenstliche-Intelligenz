import os

class DataAugmentation:
    """
    Handles all data augmentation jobs
    """

    defaultImageProcessingOptions = {
        'position': {
            'translation': {
                'active': True,
                'iterations': 100,
            },
            'rotation': {
                'active': True,
                'iterations': 180
            },
            'cropping': {
                'active': True,
                'iterations': 25
            },
            'flipping': {
                'active': True
            },
            'scaling': {
                'active': True, 
                'iterations': 25,
                'min': 0.25,
                'max': 3,
            },
            'shearing': {
                'active': True, 
                'iterations': 25,
                'maxAngle': 3,
            }             
        },
        'color': {
            'brightness': {
                'active': True,
                'iterations': 25,
                'min': 0.2,
                'max': 2
            },
            'contrast': {
                'active': True,
                'iterations': 25
            },
            'saturation': {
                'active': True,
                'iterations': 25
            },
            'hue': {
                'active': True,
                'iterations': 25
            }
        },
        'noise': {
            'gaussian': {
                'active': True,
                'iterations': 25
            },
            'saltpepper': {
                'active': True
            },
            'speckle': {
                'active': True
            }
        },
        'grayscale': True,
        'repeatForGrayscale': True
    }

    def __init__(self) -> None: 
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
        Function expects to find each different images of each class to classify in its own subdirectory and will generate new subdirectories for the new images
        """
        pass
    pass