import os
import pathlib as pl
from PIL import Image 
import cv2
import numpy as np
import random

class DataAugmentation:
    """
    Handles all data augmentation jobs
    """

    defaultImageProcessingOptions = {
        'position': {
            'translation': {
                'active': True,
                'iterations': 10,
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

    def analyzeDirectory(self, path):
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
        # Loop through all directory entries
        for entry in os.scandir(path):
            if entry.is_file():
                # Only images are interesting
                if entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    info.append({
                        'filename': entry.name,
                        'type': pl.Path(entry).suffix[1:],
                        'dimensions': Image.open(os.path.abspath(path + '\\' + entry.name)).size,
                        'size': entry.stat().st_size / 1024
                    })

        return info

    def analyzeImageDatabase(self, path='..\\images'):
        """
        Scans the entire image database for subdirectories and extracts the requested information
        Returns tuple list with the absolute path of each subdirectory and information about each image inside the folders
        """
        directories = []
        for entry in os.scandir(path):
            if entry.is_dir():
                dirPath = path + '\\' + entry.name
                directories.append((os.path.abspath(dirPath), self.analyzeDirectory(dirPath)))

        return directories

    def processImageBatch(self, batch, options=defaultImageProcessingOptions):
        """
        Processes a batch of images 
        Returns: Processed images as a list
        """
        pass

    def processImage(self, img, options=defaultImageProcessingOptions):
        """
        Processes a single image 
        Returns: Processed image
        """

        new_img = {}

        for k, v in options['position']:
            if v['active']:
                if k == 'translation':
                    n = v['iterations']
                    img.transform(img.size, Image.AFFINE, )
                elif k == 'rotation':
                    n = v['iterations']
                elif k == 'cropping':
                    n = v['iterations']
                elif k == 'flipping':
                    n = v['iterations']
                elif k == 'scaling':
                    n = v['iterations']
                elif k == 'shearing':
                    n = v['iterations']

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

    def expand2square(self, pil_img, size, background_color):  
        """
        deprecated
        """
        pil_img.thumbnail((size, size), Image.ANTIALIAS)
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
    
    def resizeAndPad(self, img, size, padColor=0):
        """
        https://stackoverflow.com/questions/44720580/resize-image-canvas-to-maintain-square-aspect-ratio-in-python-opencv
        """
        h, w = img.shape[:2]
        sh, sw = size

        # interpolation method
        if h > sh or w > sw: # shrinking image
            interp = cv2.INTER_AREA
        else: # stretching image
            interp = cv2.INTER_CUBIC

        # aspect ratio of image
        aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

        # compute scaling and pad sizing
        if aspect > 1: # horizontal image
            new_w = sw
            new_h = np.round(new_w/aspect).astype(int)
            pad_vert = (sh-new_h)/2
            pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
            pad_left, pad_right = 0, 0
        elif aspect < 1: # vertical image
            new_h = sh
            new_w = np.round(new_h*aspect).astype(int)
            pad_horz = (sw-new_w)/2
            pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
            pad_top, pad_bot = 0, 0
        else: # square image
            new_h, new_w = sh, sw
            pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

        # set pad color
        if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
            padColor = [padColor]*3

        # scale and pad
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_REPLICATE, value=padColor)

        return scaled_img

    pass


# Generate a unique list of n floats with two decimal places
def generateUniqueRandomRatios(iterations=100, ratio=1.0, decimals=2):
    ratios = []
    for i in range(iterations):
        x = round(random.uniform(-ratio, ratio), decimals)
        while x in ratios or x == 0:
            x = round(random.uniform(-ratio, ratio), decimals)
        ratios.append(x)
    return ratios

# Generates a uniform list of n floats with x decimal places
def generateUniformRatios(iterations=10, ratio=1, decimals=2):
    return [round(x, decimals) for x in np.arange(-ratio, ratio+ratio/iterations*2, ratio/iterations*2) if round(x, decimals) != 0]

# https://towardsdatascience.com/complete-image-augmentation-in-opencv-31a6b02694f5
def fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img

def generateRandomRatios(iterations=10, ratio=0.0):
    return [random.uniform(-ratio, ratio) for i in range(iterations)]

def horizontalShifts(img, ratios):
    imgs = []
    h, w = img.shape[:2]
    for ratio in ratios:
        trans_mat = np.float32([
                [1, 0, w*ratio],
                [0, 1, 0]
        ])
        imgs.append(cv2.warpAffine(img, trans_mat, img.shape[:2]))
    return imgs

def verticalShifts(img, ratios):
    imgs = []
    h, w = img.shape[:2]
    for ratio in ratios:
        trans_mat = np.float32([
                [1, 0, 0],
                [0, 1, h*ratio]
        ])
        imgs.append(cv2.warpAffine(img, trans_mat, img.shape[:2]))
    return imgs

def rotation(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

def rotations(img, steps): 
    return [rotation(img, angle) for angle in range(0, 360, int(360/steps))]

def brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value 
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def zoom(img, value):
    if value > 1 or value < 0:
        print('Value for zoom should be less than 1 and greater than 0')
        return img
    value = random.uniform(value, 1)
    h, w = img.shape[:2]
    h_taken = int(value*h)
    w_taken = int(value*w)
    h_start = random.randint(0, h-h_taken)
    w_start = random.randint(0, w-w_taken)
    img = img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    img = fill(img, h, w)
    return img

def channel_shift(img, value):
    img = img + value
    img[:,:,:][img[:,:,:]>255]  = 255
    img[:,:,:][img[:,:,:]<0]  = 0
    img = img.astype(np.uint8)
    return img

def noise_sp(image, ratio):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - ratio 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < ratio:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def noise_gauss(image, mean=0, var=0.001):
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out

def noise_speckle(image, a=0, b=1):
    gauss = np.random.normal(a, b, image.size)
    gauss = gauss.reshape(image.shape[0], image.shape[1], image.shape[2]).astype('uint8')
    noise = image + image * gauss
    return noise