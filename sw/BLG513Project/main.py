############################################################# 
# Firing Performance Measurement for                        #
# Remote Controlled Machine Gun Platform                    #
#                                                           #
# BLG 513E Image Processing Project                         #
#                                                           #
# Copyright (C)  Mehmet AKTAS 0504181576                    #
#                                                           #
#############################################################
import cv2
import math
import statistics as st
import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
from skimage.metrics import structural_similarity

# Method for finding and displaying Hough Lines
def houghLines(image):
    # Convert to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Smoothening
    imageBlur = cv2.blur(imageGray, (5,5))
    # Color correction
    imageGray = np.uint8(imageBlur*255)
    # Canny edge detection
    canny = cv2.Canny(imageGray, 100, 200)
    # Hough line transformation
    houghLines = cv2.HoughLinesP(canny, 1, np.pi / 180, 40, None, 50, 30)    
    # Make an image from Hough Lines
    height, width = canny.shape[:2]  
    houghLinesImage = np.zeros((height, width, 1), np.uint8)
    if houghLines is not None:
        for i in range(0, len(houghLines)):
            l = houghLines[i][0]
            cv2.line(houghLinesImage, (l[0], l[1]), (l[2], l[3]), (255,255,255), 3, cv2.LINE_AA)
    # Save Hough Lines image
    cv2.imwrite('DataSet/houghLinesImage.png', houghLinesImage) 

# Writes given text to the given image
# Used for user iteraction
def writeMessageToImage(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    bottomLeftCornerOfText = (500, 50)
    fontScale = 1
    fontColor = (0, 0, 0)
    cv2.putText(image, text, bottomLeftCornerOfText, font, fontScale, fontColor, thickness = 3, lineType=2)

# Method for reading user double clicks
def getPoint(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX, mouseY = x, y

# Calculates the distance of two given pixels by means of pixel value
def calculatePixelDistance(pixelFirst, pixelSecond):
    pixelDistance = math.sqrt( ((pixelFirst[0] - pixelSecond[0]) ** 2) + ((pixelFirst[1] - pixelSecond[1]) ** 2) )
    return pixelDistance   

print("Firing Performance Measurement for Remote Controlled Machine Gun Platform")

# Read background image
# Background image may be empty target or used target
fileName = 'DataSet/Fire0.png'
imageBackground = plt.imread(fileName)
# Convert to grayscale
grayBackground = cv2.cvtColor(imageBackground, cv2.COLOR_BGR2GRAY)

# Read target image after firing mission
fileName = 'DataSet/Fire1.png'
imageFire = plt.imread(fileName)
# Convert to grayscale
grayFire = cv2.cvtColor(imageFire, cv2.COLOR_BGR2GRAY)

# My purpose was to use Hough Lines Transformation to find the perpective correction points
# I could not realize my purpose with this method despite long effort
houghLines(imageBackground)
# I use the manual method and find the image perspective correction points myself
# Also this points can be asked from user
pointsSource = np.float32([[197,60],[1586,82],[1403, 992],[379,996]])
pointsDestination = np.float32([[0,0],[1600,0],[1600, 1600],[0,1600]])

# Perspective transformation matrix is calculated
M = cv2.getPerspectiveTransform(pointsSource, pointsDestination)

# Image after transformation has size of 1600x1600
transformed = np.zeros((1600, 1600), np.uint8)

# Warping the background with calculated perspective matrix
warpGrayBackground = cv2.warpPerspective(grayBackground, M, transformed.shape)
# Color correction
warpGrayBackground = (warpGrayBackground * 255).astype("uint8")

# Warping the grayscaled fire image with calculated perspective matrix
warpGrayFire = cv2.warpPerspective(grayFire, M, transformed.shape)
# Color correction
warpGrayFire = (warpGrayFire * 255).astype("uint8")

# Warping the colored fire image with calculated perspective matrix
warpFire = cv2.warpPerspective(imageFire, M, transformed.shape)
# Color correction
warpFire = (warpFire * 255).astype("uint8")

# Write the images to disc
cv2.imwrite('DataSet/grayFire.png', grayFire*255) 
cv2.imwrite('DataSet/warpGrayBackground.png', warpGrayBackground) 
cv2.imwrite('DataSet/warpGrayFire.png', warpGrayFire) 
cv2.imwrite('DataSet/warpFire.png', warpFire) 

# Find image differences between background image and fire image
(score, diff) = structural_similarity(warpGrayBackground, warpGrayFire, gaussian_weights=True, full=True, sigma=1)
# Print similarity score for information
print("Image similarity:", score)

# The diff image shows the differences between background and fire images
# Color correction
diff = (diff * 255).astype("uint8")
cv2.imwrite('DataSet/diff.png', diff) 

# Method is used for filtering noises and converting the image to binary format
thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imwrite('DataSet/thresh.png', thresh) 

# Method is used to distinguish very close bullet holes
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
cv2.imwrite('DataSet/opening.png', opening) 

# Apply Hough Circles transform on the image. 
detectedCircles = cv2.HoughCircles(opening, cv2.HOUGH_GRADIENT, 1, 8, param1 = 30, param2 = 7, minRadius = 5, maxRadius = 12) 
print("detectedCircles:", detectedCircles)

# The integer for number of circles
numberOfCircles = 0

# Draw circles that are detected. 
if detectedCircles is not None: 
    # Convert the circle parameters a, b and r to integers. 
    detectedCircles = np.uint16(np.around(detectedCircles)) 
    for pt in detectedCircles[0, :]: 
        a, b, r = pt[0], pt[1], pt[2] 
        # Draw circles on the image with red
        cv2.circle(warpFire, (a, b), r, (0, 0, 255), 2) 
        numberOfCircles += 1

print("Detected circles count:", numberOfCircles)
cv2.imwrite('DataSet/Result.png', warpFire) 

# This block of code is for user iteraction
# Firsly, the aiming point pixel is asked from user
# Than, two points of the reference squaere is asked
# Reference square edge size is 10 cm = 100 mm
userMessage = "Double click aiming point than press (a)"
userImage = warpFire.copy()
writeMessageToImage(userImage, userMessage)
cv2.imwrite('DataSet/userImage1.png', userImage)

cv2.namedWindow('User')
cv2.setMouseCallback('User', getPoint)    

# Ask necessary pixels by order 
while(1):
    cv2.imshow('User', userImage) 
    k = cv2.waitKey(0)
    if k == 27:
        break
    elif k == ord('a'):
        # First the aiming point is asked
        aimingPoint = [mouseX, mouseY]
        print("Aiming Point: " + str(mouseX) + " " + str(mouseY))
        userMessage = "Double click square top left point than press (1)"
        userImage = warpFire.copy()
        writeMessageToImage(userImage, userMessage)
        cv2.imwrite('DataSet/userImage2.png', userImage) 
    elif k == ord('1'):
        # Second, the top left corner of the reference point is asked
        squareTopLeftPoint = [mouseX, mouseY]
        print("Square Top Left Point: " + str(mouseX) + " " + str(mouseY))
        userMessage = "Double click square top right point than press (2)"
        userImage = warpFire.copy()
        writeMessageToImage(userImage, userMessage)
        cv2.imwrite('DataSet/userImage3.png', userImage) 
    elif k == ord('2'):
        # Finally, the top right corner of the reference point is asked
        squareTopRightPoint = [mouseX, mouseY]
        print("Square Top Right Point: " + str(mouseX) + " " + str(mouseY))
        break

# Calculate reference square edge distance by means of pixel
pixelDistance = calculatePixelDistance(squareTopLeftPoint, squareTopRightPoint)
print(pixelDistance)

# Calculate distance (mm) per pixel value
distancePerPixel = 100 / pixelDistance
print(distancePerPixel)

# A distance array is constructed by distances of all Hough Circle centers to aiming point
distances = []
if detectedCircles is not None: 
    for pt in detectedCircles[0, :]: 
        bulletHole = [pt[0], pt[1]]
        pixelDistance = calculatePixelDistance(aimingPoint, bulletHole)
        mmDistance = pixelDistance * distancePerPixel
        print(mmDistance)
        distances.append(mmDistance)

# Calculate and print distance mean and standard deviation
mean = st.mean(distances)
stdev = st.stdev(distances)
print("Mean(mm):" + str(mean))
print("Standard Deviation(mm):" + str(stdev))
