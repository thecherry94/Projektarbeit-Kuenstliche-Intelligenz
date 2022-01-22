# -*- coding: utf-8 -*-

import os, cv2, math
import numpy as np
from scipy.ndimage.filters import median_filter
import matplotlib.pyplot as plt
# print(cv2.__version__)

def change_contrast_brightness(img_gray, contrast=1, brightness=0):
    img_gray = img_gray.astype("float64")
    img_cb = np.maximum(np.zeros(img_gray.shape), np.minimum(np.ones(img_gray.shape)*255, 
                        contrast*img_gray+brightness)).astype("uint8")
    return img_cb


def get_pseudocolor_image(img_gray, colormap=cv2.COLORMAP_HSV):
    # openCV colormaps: COLORMAP_AUTUMN, COLORMAP_BONE, COLORMAP_JET, 
    # COLORMAP_WINTER, COLORMAP_RAINBOW, COLORMAP_OCEAN, COLORMAP_SUMMER
    # COLORMAP_SPRING, COLORMAP_COOL, COLORMAP_HSV, COLORMAP_PINK, COLORMAP_HOT
    return cv2.applyColorMap(img_gray, colormap)

def unsharp_masking(img_gray):
    img_mf = median_filter(img_gray, 1)
    lap = cv2.Laplacian(img_mf, cv2.CV_8U)
    img_sharp = np.maximum(np.zeros(img_gray.shape), np.minimum(np.ones(img_gray.shape)*255, img_gray - 0.7*lap))
    return(lap.astype("uint8"), img_sharp.astype("uint8"))

# print(os.getcwd())
parent_dir = os.getcwd()
wd = os.getcwd().split(os.sep)
if parent_dir.split(os.sep)[-1] != "ai-assignment":
    parent = os.getcwd().split(os.sep)[1:-1]
    parent_dir= "D:" + os.sep 
    for d in parent:
        parent_dir = os.path.join(parent_dir, d)
# print(parent_dir)

# print(os.path.join(os.getcwd().split(os.sep)[:-1], "data", "images", "original", "canOpener", "0abbd08fc3.jpg"))

# print(os.path.join(parent_dir, "data", "images", "original", "canOpener", "0abbd08fc3.jpg"))

# names = ["0abbd08fc3.jpg", "00f060de9a.jpg", "3bfd46aae0.jpg", "09ec4b8cc5.jpg"]

# for name in names[:]:
#     img = cv2.imread(os.path.join(parent_dir, "data", "images", "original", "canOpener", name))
    
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # meanVal = np.mean(img_gray)
#     # print(int(np.mean(img_gray)))
#     meanVal = img_gray[img_gray > 150].mean()#!= 255].mean()
#     _, img_bin = cv2.threshold(img_gray, meanVal, 255, cv2.THRESH_BINARY)
#     # img_gray[img_gray>meanVal]
    
#     cv2.namedWindow("binary image mean threshold", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("binary image mean threshold", min(len(img_bin), 1536), min(len(img_bin[0]), 864))
#     cv2.imshow("binary image mean threshold", img_bin)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
    
    
#     img_cb = change_contrast_brightness(img_gray, 2, -255) # good values: contrast 2, brightness -255
#     # lap, img_sharp = unsharp_masking(img_cb)
#     # img_pseudocolor = get_pseudocolor_image(img_gray)
    
#     img_test = cv2.GaussianBlur(img_gray,(3,3),0)
#     img_test = change_contrast_brightness(img_test, 2, -255)
#     # img1 = cv2.imread('text.png',cv2.COLOR_BGR2GRAY) # queryImage
#     # img2 = cv2.imread('original.png',cv2.COLOR_BGR2GRAY) # trainImage
#     # Initiate SIFT detector
#     orb = cv2.ORB_create()
    
#     # find the keypoints and descriptors with SIFT
#     # kp1, des1 = orb.detectAndCompute(img1,None)
#     # kp2, des2 = orb.detectAndCompute(img2,None)
#     # # create BFMatcher object
#     # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     # # Match descriptors.
#     # matches = bf.match(des1,des2)
    
#     # # Sort them in the order of their distance.
#     # matches = sorted(matches, key = lambda x:x.distance) 
#     # Draw first 10 matches.
#     # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
    
#     key_points, descriptors = orb.detectAndCompute(img_cb, None)
#     img_test = cv2.drawKeypoints(img, key_points, img_test)

#     # img_test = cv2.Laplacian(img_test,cv2.CV_64F)
#     # img_test = cv2.Sobel(img_test,cv2.CV_64F,1,0,ksize=5)  # x
#     # img_test = cv2.Sobel(img_test,cv2.CV_64F,0,1,ksize=5)  # y
    
    
#     # kernelDilEr = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#     # img_test = cv2.dilate(img_test, kernelDilEr, iterations=1)
#     # img_test = cv2.erode(img_test, kernelDilEr, iterations=1)
#     # img_diff = img_gray - img_test
#     # img_test = cv2.normalize(img_test, None, 0,255, norm_type=cv2.NORM_MINMAX)
#     # _, img_test = cv2.threshold(img_test,0,255,cv2.THRESH_OTSU)
#     # img_test = cv2.adaptiveThreshold(img_test,255, 
#     #                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#     #                                  cv2.THRESH_BINARY,11,2)
#     # img_test = cv2.GaussianBlur(img_test,(5,5),7)
    
    
    
    
#     # x = range(256)#img_gray.size)
#     # h, _ = np.histogram(img_gray, bins=256)  
#     # # print(h.shape)
#     # cum_sum = np.cumsum(h)#img_gray)#np.histogram(img_gray, bins=256)#
#     # # print(cum_sum.shape)
#     # grad = np.gradient(cum_sum)
#     # print(np.argmin(grad))
#     # # print(min(np.gradient(cum_sum)))
#     # plt.plot(x,cum_sum,label="CDF")

    
#     # print((img_gray==img_cb).all())
#     # print(img_gray.dtype, img_cb.dtype)
#     cv2.namedWindow("grayscale image", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("grayscale image", min(img_gray.shape[0], 1536), min(img_gray.shape[1], 864))
#     cv2.namedWindow("contrast brightness image", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("contrast brightness image", min(img_gray.shape[0], 1536), min(img_gray.shape[1], 864))
#     cv2.namedWindow("test", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("test", min(img_gray.shape[0], 1536), min(img_gray.shape[1], 864))
#     # cv2.namedWindow("laplacian", cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow("laplacian", min(img_gray.shape[0], 720), min(img_gray.shape[1], 480))
#     # cv2.namedWindow("sharp image", cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow("sharp image", min(img_gray.shape[0], 720), min(img_gray.shape[1], 480))

#     # cv2.namedWindow("pseudocolor image", cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow("pseudocolor image", 720, 480)
    
#     cv2.imshow("grayscale image", img_gray)
#     cv2.imshow("contrast brightness image", img_cb)
#     cv2.imshow("test", img_test)#abs(img_test-img_gray))
#     # cv2.imshow("laplacian", lap)
#     # cv2.imshow("sharp image", img_sharp)
#     # cv2.imshow("pseudocolor image", img_pseudocolor)
    
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



