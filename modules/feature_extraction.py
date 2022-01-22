# -*- coding: utf-8 -*-
"""
This file provides functions to extract features out of object images for 
classification and save them in a csv file.
"""
from skimage.transform import hough_line, hough_line_peaks
import numpy as np
import cv2, imutils, math
from imutils import perspective
import modules.img_manipulations as im

images = dict()

# ??? OPTIMIZE does not work like that; maybe change? 
# def angles(imgPath):
#     image = cv2.imread(imgPath)
#     display_image(image, title="original image")

#     # Compute arithmetic mean
#     image = np.mean(image, axis=2)
#     display_image(image, title="mean image")
    
#     # Perform Hough Transformation to detect lines
#     hspace, angles, distances = hough_line(image)
#     print("hough line angles", angles)
    
#     # Find angle
#     angle=[]
#     for _, a , distances in zip(*hough_line_peaks(hspace, angles, distances)):
#         angle.append(a)
#     print("hough line peaks angles", a)
    
#     # Obtain angle for each line
#     angles = [a*180/np.pi for a in angle]
    
#     # Compute difference between the two lines
#     angle_difference = np.max(angles) - np.min(angles)
#     print(angle_difference)


# def sift_corner_detection(img):
#     fast = cv2.FastFeatureDetector_create()
#     # find and draw the keypoints
#     fast.setNonmaxSuppression(0)
#     kp = fast.detect(img,None)
    
#     keypoints = []
#     for k in kp:
#         if k.class_id != -1:
#             keypoints.append(k)
            
#     # mask =  [(k.class_id != -1) for k in kp]
#     # kp = kp[mask]
#     if len(keypoints)>0:
#         print(keypoints[0].class_id, keypoints[0].pt)
#     img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
#     return img2

# # OPTIMIZE too many corners, not all relevant contours
# def harris_corner_detection(img):#, path):#Path):
#     # imgOrig = cv2.imread(path)
#     # imgPrep = prepared_image(img)
#     cntrs = find_contours(img)#Prep)#, threshold=50)
#     imgBlank = np.zeros_like(img)#Prep)
#     imgCntr = cv2.drawContours(imgBlank, cntrs, -1, (255,255,255), 1)
    
#     # display_image(img, title="original image", destroy=False)
#     # display_image(imgPrep, title="prepared image", destroy=False)
#     # display_image(imgCntr, title="contour only", destroy=False)
    
#     imgDst =  np.float32(imgCntr)
#     dst = cv2.cornerHarris(imgDst,2,3,0.04)
    
#     # result is dilated for marking the corners, not important
#     # dst = cv2.dilate(dst,None)
#     # print("dst shape: ", dst.shape)
#     # print("dst>0.02*dst.max(): ", dst>0.02*dst.max())
#     # print("sum(dst>0.02*dst.max())", sum(dst>0.02*dst.max()))
#     # print()
#     # Threshold for an optimal value, it may vary depending on the image.
#     numCorners = np.count_nonzero((dst>0.5*dst.max()))
#     # dst = cv2.dilate(dst,None)
#     # imgOrig[dst>0.5*dst.max()]=[0,0,255]
#     # display_image(imgOrig, title="corner image")
  
#     return numCorners#sum(isCorner)#(dst>0.02*dst.max()).flatten())


def num_corners(img, detector="shi-tomasi"):
    # cntrs = find_contours(img, threshold=100)
    # imgBlank = np.zeros_like(img)
    # imgCntr = cv2.drawContours(imgBlank, cntrs, -1, (255,255,255), 1)
    
    numCorners = -1
    if detector == "shi-tomasi":
        corners = cv2.goodFeaturesToTrack(img, 0, 0.01, 30)#25,0.01,10)
        corners = np.int0(corners)
        numCorners = len(corners)
    elif detector == "harris":
        corners = cv2.goodFeaturesToTrack(img, 0, 0.01, 30, useHarrisDetector=True, k=0.04)
        corners = np.int0(corners)
        numCorners = len(corners)
     
    imgCorners = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(imgCorners,(x,y),3,255,-1)
    # display_image(imgCorners, title=detector + " corner detection")
    images[detector] = imgCorners
    
    return numCorners
    

def max_area_contour(img, cntrs=None):#Path):
    """  
    Parameters
    ----------
    imgPath : str
        Path to the image.

    Returns
    -------
    maxCntr : Array of int32
        The contour with the largest area.
    """
    # maxCntr = max(find_contours(imgPath), key=cv2.contourArea)
    if cntrs is None:
        maxCntr = max(find_contours(img), key=cv2.contourArea)
    else:
        maxCntr = max(cntrs, key=cv2.contourArea)
    return maxCntr
 
    
def prepared_image(img):
    if type(img) is str:
        # print("in if img str")
        img = cv2.imread(img)
    # print("image shape in prepared image", img.shape, type(img))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGray = im.change_contrast_brightness(imgGray, 2, -255) # good values: contrast 2, brightness -255
    
    images["original"] = img
    
    
    imgGray = cv2.normalize(imgGray,  imgGray, 0, 255, cv2.NORM_MINMAX)
    images["prepared"] = imgGray
    
    
    # imgBinary = cv2.threshold(imgGray,,255,cv.THRESH_BINARY)
    
    # # ??? OPTIMIZE reduce shadows 
    # imgTransf = cv2.dilate(imgGray, np.ones((9,9), np.uint8))
    # imgTransf = cv2.medianBlur(imgTransf, 21)
    # imgTransf = 255 - cv2.absdiff(imgTransf, imgGray)
    
    return imgGray
    
    
def find_contours(img, threshold=None):
    """  
    Parameters
    ----------
    imgPath : str
        Path to the image.
    threshold: int
        Ignore contours with areas smaller than or equal to this value. If it \
            is None every contour will be returned. Suggested value: 100.

    Returns
    -------
    cntrs : list
        The contours in the image. If a threshold is given only the contours \
            with areas larger than that threshold are returned.
    """
    # # load image and convert to grayscale
    # # TODO handle unsupported file formats 
    # img = cv2.imread(imgPath)
    # imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # # ??? OPTIMIZE reduce shadows 
    # imgTransf = cv2.dilate(imgGray, np.ones((9,9), np.uint8))
    # imgTransf = cv2.medianBlur(imgTransf, 21)
    # imgTransf = 255 - cv2.absdiff(imgTransf, imgGray)
    
    # if type(img) is str:
    #     img = cv2.imread(img)
    #     # prepare image for feature extraction
    #     img = prepared_image(img)

    # perform Canny edge detection
    # ??? parameters
    
    # display_image(img, title="prepared image")
    
    imgTransf = cv2.GaussianBlur(img, (3,3), 0)
    imgTransf = cv2.Canny(img, 100, 200)#50, 100)
    images["canny"] = imgTransf
    
    # dilation and erosion to close edge gaps
    kernelDilEr = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    imgTransf = cv2.dilate(imgTransf, kernelDilEr, iterations=1)
    imgTransf = cv2.erode(imgTransf, kernelDilEr, iterations=1)
    # display_image(imgTransf, title="canny edge detection with closed gaps")
    images["canny closed gaps"] = imgTransf
    
    # find contours in the edge map
    cntrs, _ = cv2.findContours(imgTransf.copy(), cv2.RETR_EXTERNAL, 
                                 cv2.CHAIN_APPROX_SIMPLE)
    
    if threshold:
        # only use contours with an area larger than the threshold 
        areas = np.array([cv2.contourArea(c) for c in cntrs])
        cntrs = np.array(cntrs)[areas>threshold]
    
    imgContours = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    imgContours = cv2.drawContours(imgContours, cntrs, -1, (0,0,255), 1)
    images["contours"] = imgContours
    # display_image(imgContours, title="contours")
    
    return cntrs
 
    
def aspect_ratio(img):#Path):
    """
    Parameters
    ----------
    imgPath : str
        Path to the image.

    Returns
    -------
    ratio : list
        The aspect ratio of the minumum area bounding box of the maximum area
        contour. It is always the short side of the box divided by the long 
        side.
    """
    maxCntr = max_area_contour(img)#Path)
    
    imgContour = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    imgContour = cv2.drawContours(imgContour, maxCntr, -1, (0,0,255), 2)
    images["max area contour"] = imgContour
    
    # compute rotated bouning box and order points: top-left, top-right, 
    # bottom-right, bottom-left
    box = cv2.minAreaRect(maxCntr)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    
    lengths = (math.dist(box[0], box[1]), math.dist(box[0], box[3]))
    ratio = min(lengths)/max(lengths)
    
    return ratio



def perimeter_area_ratio(img):
    # cntrs = find_contours(img, threshold=100)
    maxCntr = max_area_contour(img)#, cntrs)
    
    perimeter = cv2.arcLength(maxCntr, True)
    area = cv2.contourArea(maxCntr)
    
    return perimeter/area
    
    # perimeter = 0
    # for c in cntrs:
    #     perimeter += cv2.arcLength(c, True)
        
    # maxPerimeter = cv2.arcLength(maxCntr, True)
    
    # return maxPerimeter/perimeter


def display_image(img, title="image", destroy=True):
    if type(img) is str:
        img = cv2.imread(img)
    # img = cv2.resize(img, (960, 540))
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, min(img.shape[0], 1536), min(img.shape[1], 864))
    cv2.imshow(title, img)
    cv2.waitKey(0)
    if destroy:
        cv2.destroyAllWindows()
    