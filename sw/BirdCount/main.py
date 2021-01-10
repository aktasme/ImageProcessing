############################################################# 
# Bird Counting Algorithm                                   #
#                                                           #
# BLG 513E Image Processing HW#2                            #
#                                                           #
# Copyright (C)  Mehmet AKTAS 0504181576                    #
#                                                           #
#############################################################
import cv2
import math
import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 

# Convert rgb image to grayscale
# Formulation: Gray = 0.21 Red + 0.72 Green + 0.07 Blue
def rgb2gray(img):
    # Already GreyScale
    if(len(img.shape)<3):
        return img
    # Convert to grayscale
    H,W = img.shape[:2]
    gray = np.zeros((H,W), np.uint8)
    for i in range(H):
        for j in range(W):
            gray[i,j] = np.clip(0.07 * img[i,j,0] + 0.72 * img[i,j,1] + 0.21 * img[i,j,2], 0, 255)    
    return gray

# K Means Clustering algorithm for calculating threshold
# Threshold value is strengthened after K Means
# Algorithm taken from course lectures (UCL-L1_Segmentation_02.pdf)
def kMeansClustering(img):
    threshold = 0
    newThreshold = 128
    counter = 0
    # Continue until the threshold value is 0.1 near from the previous one
    while(abs(threshold - newThreshold) > 0.1):
        # Initialize sum and count values
        threshold = newThreshold
        foregroundSum = 0
        backgroundSum = 0
        foregroundCount = 0
        backgroundCount = 0
        H,W = img.shape[:2]
        for i in range(H):
            for j in range(W):
                pixel = img[i,j]
                if(pixel >= threshold):
                    foregroundCount += 1
                    foregroundSum += pixel
                else:
                    backgroundCount += 1
                    backgroundSum += pixel
        foregroundMean = foregroundSum / foregroundCount
        backgroundMean = backgroundSum / backgroundCount
        newThreshold = (foregroundMean + backgroundMean) / 2
    # This formulation is used to strengthen the threshold
    # I have found the formulation with some iterations
    newThreshold = newThreshold + ( 1.95 * (newThreshold - 128))
    return newThreshold

# Calculated threshold is applied to an image
def thresholding(img, threshold):
    H,W = img.shape[:2]
    for i in range(H):
        for j in range(W):
            if(img[i,j] >= threshold):
                img[i,j] = 255
            else:
                img[i,j] = 0
    return img

# Converts pixel values according to background decision
# Counts zero(0x00) and one(0xFF) pixels in the image
# Reverts zeros and ones if needed
def findBackground(img):
    zeros = 0
    ones = 0
    H,W = img.shape[:2]
    for i in range(H):
        for j in range(W):
            if(img[i,j] == 0):
                zeros += 1
            else:
                ones += 1
    # Convert foreground and background pixels
    if(ones > zeros):
        for i in range(H):
            for j in range(W):
                if(img[i,j] == 0):
                    img[i,j] = 255
                else:
                    img[i,j] = 0
    return img

# Morphological Dilation with 3x3 Square Structuring Element
def dilation(img):
    # 3x3 Square Structuring Element
    seSquare = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
    H,W = img.shape[:2]
    dilationImage = np.zeros([H, W], dtype=np.uint8)
    for i in range(2, H-2, 1):
        for j in range(2, W-2, 1):
            # If image pixel is white, make other pixels white with a neighborhood of structuring element
            if(img[i,j] == 255):
                for offset in seSquare:
                    x = i + offset[1]
                    y = j + offset[0]
                    if(dilationImage[x,y] == 0):
                        dilationImage[x,y] = 255            
    return dilationImage    

# Morphological Erosion with 3x3 Square Structuring Element
def erosion(img):
   # 3x3 Square Structuring Element
    seSquare = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
    H,W = img.shape[:2]
    erosionImage = np.zeros([H, W], dtype=np.uint8)
    for i in range(2, H-2, 1):
        for j in range(2, W-2, 1):
            overlap = True
            # If all pixels are overlapping with a neighborhood of structuring element make the center pixel white
            for offset in seSquare:
                x = i + offset[1]
                y = j + offset[0]
                if(img[x,y] == 0):
                    overlap = False
                    break
            if(overlap == True):
                erosionImage[i,j] = 255
    return erosionImage    

# Lables the neighbours of given pixel with given label counter
# This function works recursively
# Algorithm taken from course lectures (UCL-L1_Segmentation_02.pdf)
def label(xStart, yStart, counter, image, labeledImage):
    # Connectivity matrix of 4-components neighborhood 
    connectivityMatrix = [[-1, 0], [0, -1], [0, 1], [1, 0]]
    labeledImage[xStart, yStart] = counter;
    H,W = image.shape[:2]
    for offset in connectivityMatrix:
        x = xStart + offset[1]
        y = yStart + offset[0]
        if(x < 0 or x >= H or y < 0 or y >= W):
            continue
        if(image[x,y] == 255 and labeledImage[x,y] == 0):
            label(x, y, counter, image, labeledImage)

# Find the connected components using label function with 4-components neighborhood
# Algorithm taken from course lectures (UCL-L1_Segmentation_02.pdf)
def connectedComponents(image):
    H,W = image.shape[:2]
    labeledImage = np.zeros([H, W], dtype=np.uint16)
    counter = 1
    for i in range(H):
        for j in range(W):
            if(image[i,j]==255 and labeledImage[i,j]==0):
                label(i, j, counter, image, labeledImage)
                counter += 1
    return counter, labeledImage

print("Finding number of birds in images")

# Read image (bird_1.jpg, bird_2.jpg, bird_3.bmp)
fileName = 'bird_1.jpg'
image = plt.imread(fileName)

# Displays image
plt.subplot(3, 2, 1), plt.imshow(image, 'gray')
plt.title('Original')
plt.xticks([]), plt.yticks([])

# Convert to grayscale
image = rgb2gray(image)

# Displays image
plt.subplot(3, 2, 2), plt.imshow(image, 'gray')
plt.title('Grayscale')
plt.xticks([]), plt.yticks([])

# K Means Clustering algorithm for calculating threshold
threshold = kMeansClustering(image)
print('Threshold:' + str(threshold))

# Calculated threshold is applied to an image
image = thresholding(image, threshold)

# Displays image
plt.subplot(3, 2, 3), plt.imshow(image, 'gray')
plt.title('Thresholding')
plt.xticks([]), plt.yticks([])

# Converts pixel values according to background decision
image = findBackground(image)

# Displays image
plt.subplot(3, 2, 4), plt.imshow(image, 'gray')
plt.title('Background Correction')
plt.xticks([]), plt.yticks([])

# Morphological Opening (Dilation and Erosian)
image = erosion(image)
image = dilation(image)

# Displays image
plt.subplot(3, 2, 5), plt.imshow(image, 'gray')
plt.title('Morphological Opening')
plt.xticks([]), plt.yticks([])

# Finding connected components with 4-components neighborhood
ret, labels = connectedComponents(image)

# One of the label is background
# Substract 1 to find the number of birds
numberOfBirds = ret - 1

# For assigning different colors to each bird
# OpenCV functions are used for generating colored images
# This block is taken from the internet
labelHue = np.uint8(200 * labels / np.max(labels))
blankChannel = 255 * np.ones_like(labelHue)
image = cv2.merge([labelHue, blankChannel, blankChannel])
image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
image[labelHue == 0] = 0

# Displays image
plt.subplot(3, 2, 6), plt.imshow(image, 'gray')
plt.title('Connected Components\nNumber of Birds: ' + str(numberOfBirds))
plt.xticks([]), plt.yticks([])

plt.savefig('Result.jpg')

print('Number of Birds: ' + str(numberOfBirds))
print('Result image recorded to Result.jpg')

# Displays subplotted images
plt.show()
