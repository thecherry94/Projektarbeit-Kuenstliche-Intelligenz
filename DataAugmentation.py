import os
import cv2
import numpy as np
import random
import pickle

def trainTestSplit(data, labels, ratio):
    return ((data[:int(len(data)*ratio)], labels[:int(len(data)*ratio)]), (data[int(len(data)*ratio):], labels[int(len(data)*ratio):]))

def loadAndPrepareAugmentedData():
    dataset = np.load(r'data/images/augmented/augmentation.npy', allow_pickle=True)

    images = []
    labels = []

    for idx, d in enumerate(dataset):
        images.extend(d)
        for e in d:
            labels.append([idx])
    
    return (np.array(images), np.array(labels))


def loadAndPrepareOriginalData():
    original = np.load(r'data/images/augmented/original.npy', allow_pickle=True)

    original_images = []
    original_labels = []

    for idx, d in enumerate(original):
        original_images.extend(d)
        for e in d:
            original_labels.append([idx])

    return (np.array(original_images), np.array(original_labels))

def process(image_size=32):
    relImgPath = os.path.join("data", "images", "original")
    imgPaths = []

    grandlist = [[], [], [], []]
    originalList = [[], [], [], []]
    idxgrand = 0

    for dirpath, dirnames, filenames in os.walk(relImgPath):     
        if dirnames:          
            classes = {}
            for index, name in enumerate(dirnames):
                classes[name]=index
        for filename in filenames:# [f for f in filenames if f.endswith(suportedImgFomats)]:
            path = os.path.join(dirpath, filename)
            augpath = path.replace("original", "augmented")

            image = cv2.imread(path, cv2.COLOR_BGR2RGB)
            auglist = []
            image = resizeAndPad(image, (image_size, image_size))
            originalList[idxgrand].append(image)
            auglist.append(cv2.flip(image, 0))
            auglist.append(cv2.flip(image, 1))
            #ratios = generateUniqueRandomRatios(15, 0.5)
            
            numRotations = 6
            rotlist = rotations(image, numRotations)

            for rot in rotlist:
                auglist.append(rot)
                originalList[idxgrand].append(rot)
                shiftRatios = generateUniqueRandomRatios(5, 0.4)
                brightnessRatios = generateUniqueRandomRatios(10, 0.75)
                saltPepperNoiseRatios = generateUniqueRandomRatios(8, 0.25, decimals=3, onlyPositive=True)
                zoomRatios = [0.5 for i in range(8)] #generateUniqueRandomRatios(10, 0.8, decimals=2, onlyPositive=True)
                channelShiftRatios = generateUniqueRandomRatios(10, 0.15, decimals=3, onlyPositive=True)
                speckleNoiseRatios = generateUniqueRandomRatios(3, 0.05, decimals=5, onlyPositive=True)

                auglist.extend(verticalShifts(rot, shiftRatios))
                auglist.extend(horizontalShifts(rot, shiftRatios))

                for ratio in brightnessRatios:
                    auglist.append(brightness(rot, ratio))
                
                for ratio in saltPepperNoiseRatios:
                    auglist.append(noise_sp(rot, ratio))

                for ratio in zoomRatios:
                    auglist.append(zoom(rot, ratio))

                for ratio in channelShiftRatios:
                    auglist.append(channel_shift(rot, ratio))

                for ratio in speckleNoiseRatios:
                    auglist.append(noise_speckle(rot, ratio))              

                grandlist[idxgrand].extend(auglist)
                
        if len(filenames) > 0:
            grandlist[idxgrand] = np.array(grandlist[idxgrand])
            originalList[idxgrand] = np.array(originalList[idxgrand])
            idxgrand += 1

    grandlist = np.array(grandlist)
    originalList = np.array(originalList)
    with open(r'data/images/augmented/augmentation.npy', 'wb') as f:
        pickle.dump(grandlist, f, protocol=4)
    #np.save(r'data/images/augmented/augmentation.npy', grandlist, allow_pickle=True)
    np.save(r'data/images/augmented/original.npy', originalList, allow_pickle=True)                          

def resizeAndPad(img, size, padColor=0):
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

# Generate a unique list of n floats with two decimal places

def generateUniqueRandomRatios(iterations=10, ratio=1.0, decimals=2, onlyPositive=False):
    ratios = []
    for i in range(iterations):
        x = round(random.uniform(0 if onlyPositive else -ratio, ratio), decimals)
        while x in ratios or x == 0:
            x = round(random.uniform(0 if onlyPositive else -ratio, ratio), decimals)
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