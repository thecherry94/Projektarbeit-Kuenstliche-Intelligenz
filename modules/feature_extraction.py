# -*- coding: utf-8 -*-
"""
This file provides functions to extract features out of object images for 
classification and save them in a csv file.
"""
import math
import os

import numpy as np
import pandas as pd

import cv2 
from imutils import perspective


images = dict()


def prepared_image(img):
    """
    Prepares the image for feature extraction by converting to grayscale, 
    changing contrast and brightness, and normalizing.

    Parameters
    ----------
    img : str or numpy.ndarray
        The (grayscale) image array to be changed or its path.

    Returns
    -------
    img_gray : numpy.ndarray
        The prepared image array.

    """
    if type(img) is str:
        img = cv2.imread(img)
    
    # Convert to grayscale, change contrast/brightness, and normalize image.
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = change_contrast_brightness(img_gray, 2, -255)
    img_gray = cv2.normalize(img_gray, img_gray, 0, 255, cv2.NORM_MINMAX)
    
    # Save the original and prepared image in global dict 'images'.
    images["original"] = img
    images["prepared"] = img_gray
    
    return img_gray


def change_contrast_brightness(img_gray, contrast=1., brightness=0.):
    """
    Change an image's contrast and brightness.

    Parameters
    ----------
    img_gray : numpy.ndarray
        The (grayscale) image array to be changed.
    contrast : float, optional
        Factor for changing the contrast.
    brightness : float, optional
        Summand for changing the brightness.

    Returns
    -------
    img_cb : numpy.ndarray
        The adapted image array.

    """
    
    # Convert to float for image manipulation.
    img_gray = img_gray.astype("float64")
    
    img_cb = contrast*img_gray+brightness
    
    # Make sure the new image's values are in range [0,255] and convert back
    # to unigned integer.
    img_max255 = np.minimum(np.ones(img_gray.shape)*255, img_cb)
    img_cb = np.maximum(np.zeros(img_gray.shape), img_max255).astype("uint8")
    
    return img_cb


def aspect_ratio(img):
    """
    Calculates the aspect ratio of the biggest contour in the image.

    Parameters
    ----------
    img : numpy.ndarray
        The (grayscale) image array.

    Returns
    -------
    float
        The aspect ratio of the minimum area bounding box of the maximum area
        contour. It is always the short side of the box devided by the long
        side.

    """
    
    max_cntr = max_area_contour(img)
    
    # Draw max area contour and save in global dict 'images'
    img_cntr = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_cntr = cv2.drawContours(img_cntr, max_cntr, -1, (0,0,255), 1)
    images["max area contour"] = img_cntr
    
    # Calculate the minimum bounding box's corner coordinates
    box = cv2.minAreaRect(max_cntr)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    
    # Calculate the bounding box's width and height
    lengths = (math.dist(box[0], box[1]), math.dist(box[0], box[3]))
    
    return min(lengths)/max(lengths)


def num_corners(img, detector="shi-tomasi"):
    """
    Calculates the number of corners found in an imamge optionally using Harris 
    Corner Detection or Shi-Tomasi Corener Detection.

    Parameters
    ----------
    img : numpy.ndarray
        The (grayscale) image array.
    detector : str, optional
        The name for the chosen detector. Possible options are "shi-tomasi" and 
        "harris". The default is "shi-tomasi".

    Returns
    -------
    num_corners : int
        The number of corners detected in the image.

    """
    
    num_corners = -1
    if detector == "shi-tomasi":
        corners = cv2.goodFeaturesToTrack(img, 
                                          maxCorners=0, 
                                          qualityLevel=0.01, 
                                          minDistance=30)
        corners = np.int0(corners)
        num_corners = len(corners)
    elif detector == "harris":
        corners = cv2.goodFeaturesToTrack(img, 0, 0.01, 30, 
                                          useHarrisDetector=True, 
                                          k=0.04)
        corners = np.int0(corners)
        num_corners = len(corners)
    
    # Draw the detected corners in the image and save in global dict images.
    img_corners = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(img_corners,(x,y),3,255,-1)
    images[detector] = img_corners
    
    return num_corners
 

def perimeter_area_ratio(img):
    """
    Calculate the area between the maximum area contour's perimeter and area.

    Parameters
    ----------
    img : numpy.nd_array
        The (grayscale) image array.

    Returns
    -------
    float
        The perimeter-area ratio.

    """
    
    max_cntr = max_area_contour(img)
    
    perimeter = cv2.arcLength(max_cntr, True)
    area = cv2.contourArea(max_cntr)
    
    return perimeter/area
   

def max_area_contour(img, cntrs=None):
    """ 
    Determine the contour with the biggest area.
    
    Parameters
    ----------
    img : numpy.ndarray
        The (grayscale) image array.
    cntrs : numpy.ndarray, optional
        All the contours cound in the image.

    Returns
    -------
    maxCntr : numpy.ndarray
        The contour with the largest area as a vector of points.
    """
    if cntrs is None:
        max_cntr = max(find_contours(img), key=cv2.contourArea)
    else:
        max_cntr = max(cntrs, key=cv2.contourArea)
    return max_cntr
    
    
def find_contours(img, threshold=None):
    """
    Find contours in an image.

    Parameters
    ----------
    img : numpy.ndarray
        The (grayscale) image array.
    threshold : int, optional
        Ignore contours with areas smaller than or equal to this value. If it
        is None every contour will be returned. Suggested value: 100. The 
        default is None.

    Returns
    -------
    cntrs : tuple
        The contours in the image. If a threshold is given only the contours
        with areas larger than that threshold are returned.

    """
    
    # Gaussian Blurring as preparation for... 
    img_transf = cv2.GaussianBlur(img, (3,3), 0)
    # ... Canny edge detection
    img_transf = cv2.Canny(img, 100, 200)#50, 100)
    
    # Save the transformed image (edge map) in global dict 'images'
    images["canny"] = img_transf
    
    # Dilation and erosion to close edge gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img_transf = cv2.dilate(img_transf, kernel, iterations=1)
    img_transf = cv2.erode(img_transf, kernel, iterations=1)
    
    # Save the closed-edge-gaps image in global dict 'images'
    images["canny closed gaps"] = img_transf
    
    # Find contours in the edge map
    # RETR_EXTERNAL --> find only the extreme outer contours
    # CHAIN_APPROX_SIMPLE --> only saves the end points of line segments
    cntrs, _ = cv2.findContours(img_transf.copy(), 
                                mode=cv2.RETR_EXTERNAL,
                                method=cv2.CHAIN_APPROX_SIMPLE)
    
    if threshold:
        # only use contours with an area larger than the threshold 
        areas = np.array([cv2.contourArea(c) for c in cntrs])
        cntrs = np.array(cntrs)[areas>threshold]
    
    # Draw contour points on the image and save in global dict 'images'
    imgContours = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    imgContours = cv2.drawContours(imgContours, cntrs, -1, (0,0,255), 1)
    images["contours"] = imgContours
    
    return cntrs
 

def display_image(img, title="image", destroy=True):
    """
    Displays an image in a separate window.

    Parameters
    ----------
    img : numpy.ndarray or str
        The (grayscale) image array to be displayed or its path.
    title : str, optional
        The window's title. The default is "image".
    destroy : bool, optional
        Determines whether the window will be automatically destroyed after 
        pressing a button on the keyboard. The default is True.

    Returns
    -------
    None.

    """
    
    if type(img) is str:
        img = cv2.imread(img)
        
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, min(img.shape[0], 1536), min(img.shape[1], 864))
    cv2.imshow(title, img)
    cv2.waitKey(0)
    
    if destroy:
        cv2.destroyAllWindows()
        
        
def extract_features(imgpaths, classes, display_imgs=False, imgs2show=["original", "prepared", "canny", "canny closed gaps", "max area contour", "harris", "shi-tomasi"], show_all=False): 
    global images
    
    features = ["Relative Image Path",
           "Class Name",
           "Class Index",
           "Aspect Ratio",
           "Number of Corners (Harris)",
           "Number of Corners (Shi-Tomasi)",
           "Perimeter Area Ratio"]
    df = pd.DataFrame(columns=features)
    
    show_img = 1
    if not show_all:
        show_img = len(imgpaths)//3 
    
    for i, path in enumerate(imgpaths):
        # Extract features and save them in a row of DataFrame df
        c = path.split(os.sep)[-2]
        img = prepared_image(path)
        row = []
        row.append(path)
        row.append(c) 
        row.append(float(classes[c]))
        row.append(aspect_ratio(img)) 
        row.append(float(num_corners(img, detector="harris")))
        row.append(float(num_corners(img, detector="shi-tomasi")))
        row.append(perimeter_area_ratio(img))
        row = pd.Series(row, index=features)
        df = df.append(row, ignore_index=True)
        
        # User feedback about the progress
        print(str(format((100./len(imgpaths))*i, ".2f"))+" %", end="\r")
        
        # Potentially show the images in different stages of the extraction progress
        if display_imgs:
            if not i%show_img:
                for img_name in imgs2show[:-1]:
                    display_image(images[img_name], title=img_name, destroy=False)
                display_image(images[imgs2show[-1]], title=imgs2show[-1])
                
    return df
    