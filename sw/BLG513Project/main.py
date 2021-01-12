############################################################# 
# Firing Performance Measurement for                        #
# Remote Controlled Machine Gun Platform                    #
#                                                           #
# BLG 513E Image Processing Project                         #
#                                                           #
# Copyright (C)  Mehmet AKTAS 0504181576                    #
#                                                           #
#############################################################
from skimage.metrics import structural_similarity
import cv2
import math
import statistics as st
import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 

def findLines(image):
    fileName = 'DataSet/Fire6.png'
    imageA = plt.imread(fileName)
    
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageBlur = cv2.blur(imageGray, (5,5))
    #cv2.imshow("imageGray1", imageGray)
    imageGray = np.uint8(imageBlur*255)
    #cv2.imshow("imageGray2", imageGray)
    # Canny edge detection
    dst = cv2.Canny(imageGray, 100, 200)
    #cv.Canny(src, 50, 200, None, 3)
    #cv2.imshow("canny", dst)
    #cv2.waitKey(0)

    # Copy edges to the images that will display the results in BGR     
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    #cdstP = np.copy(cdst)
    height, width = cdst.shape[:2]  
    cdstP = np.zeros((height, width, 1), np.uint8)
        
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 40, None, 50, 30)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (255,255,255), 3, cv2.LINE_AA)

    #cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

def writeMessageToImage(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    bottomLeftCornerOfText = (600, 50)
    fontScale = 1
    fontColor = (0, 0, 0)
    cv2.putText(image, text, bottomLeftCornerOfText, font, fontScale, fontColor, thickness = 3, lineType=2)

def getPoint(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX, mouseY = x, y

def calculatePixelDistance(pixelFirst, pixelSecond):
    pixelDistance = math.sqrt( ((pixelFirst[0] - pixelSecond[0]) ** 2) + ((pixelFirst[1] - pixelSecond[1]) ** 2) )
    return pixelDistance   

print("Firing Performance Measurement for Remote Controlled Machine Gun Platform")

fileName = 'DataSet/Fire0.png'
imageBackground = plt.imread(fileName)
grayBackground = cv2.cvtColor(imageBackground, cv2.COLOR_BGR2GRAY)

#findLines(imageBackground)

fileName = 'DataSet/Fire1.png'
imageFire = plt.imread(fileName)
grayFire = cv2.cvtColor(imageFire, cv2.COLOR_BGR2GRAY)

pointsSource = np.float32([[197,60],[1586,82],[1403, 992],[379,996]])
#pointsDestination = np.float32([[197,60],[1586,82],[1586, 1450],[197,1472]])
pointsDestination = np.float32([[0,0],[1600,0],[1600, 1600],[0,1600]])

M = cv2.getPerspectiveTransform(pointsSource, pointsDestination)

transformed = np.zeros((1600,1600), np.uint8)

warpGrayBackground = cv2.warpPerspective(grayBackground, M, transformed.shape)
#cv2.imshow("warpGrayBackground", warpGrayBackground) 
warpGrayBackground = (warpGrayBackground * 255).astype("uint8")

warpGrayFire = cv2.warpPerspective(grayFire, M, transformed.shape)
#cv2.imshow("warpGrayFire", warpGrayFire)
warpGrayFire = (warpGrayFire * 255).astype("uint8")

warpFire = cv2.warpPerspective(imageFire, M, transformed.shape)
#cv2.imshow("warpFire", warpFire)
warpFire = (warpFire * 255).astype("uint8")

findLines(warpFire)

cv2.imwrite('DataSet/grayFire.png', grayFire*255) 
cv2.imwrite('DataSet/warpGrayBackground.png', warpGrayBackground) 
cv2.imwrite('DataSet/warpGrayFire.png', warpGrayFire) 
cv2.imwrite('DataSet/warpFire.png', warpFire) 

(score, diff) = structural_similarity(warpGrayBackground, warpGrayFire, gaussian_weights=True, full=True, sigma=1)
print("Image similarity:", score)

# The diff image contains the actual image differences between the two images
# and is represented as a floating point data type in the range [0,1] 
# so we must convert the array to 8-bit unsigned integers in the range
# [0,255] before we can use it with OpenCV
diff = (diff * 255).astype("uint8")

cv2.imwrite('DataSet/diff.png', diff) 

#diff = cv2.medianBlur(diff, 7)

# Threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 111, 3)

cv2.imwrite('DataSet/thresh.png', thresh) 

# To filling circles
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

cv2.imwrite('DataSet/opening.png', opening) 

#erosion = cv2.erode(thresh, kernel, iterations = 2)
#dilation = cv2.dilate(thresh, kernel, iterations = 2)

#cv2.imshow('diff', diff)
#cv2.imshow('thresh', thresh)
#cv2.imshow('closing', closing)
#cv2.imshow('erosion', erosion)
#cv2.imshow('dilation', dilation)


# Apply Hough transform on the blurred image. 
#detectedCircles = cv2.HoughCircles(closing, cv2.HOUGH_GRADIENT, 1, 6, param1 = 30, param2 = 7, minRadius = 0, maxRadius = 8) 
detectedCircles = cv2.HoughCircles(opening, cv2.HOUGH_GRADIENT, 1, 8, param1 = 30, param2 = 7, minRadius = 5, maxRadius = 12) 
print("detected_circles:", detectedCircles)

# Draw circles that are detected. 
if detectedCircles is not None: 
    # Convert the circle parameters a, b and r to integers. 
    detectedCircles = np.uint16(np.around(detectedCircles)) 
    for pt in detectedCircles[0, :]: 
        a, b, r = pt[0], pt[1], pt[2] 
        # Draw the circumference of the circle. 
        cv2.circle(warpFire, (a, b), r, (0, 0, 255), 2) 
        # Draw a small circle (of radius 1) to show the center. 
        #cv2.circle(imageFire, (a, b), 1, (0, 0, 255), 3) 


cv2.imwrite('DataSet/Result.png', warpFire) 

userMessage = "Double click aiming point than press (a)"
userImage = warpFire.copy()
writeMessageToImage(userImage, userMessage)

cv2.namedWindow('result')
cv2.setMouseCallback('result', getPoint)    

while(1):
    cv2.imshow('result', userImage)
    k = cv2.waitKey(0)
    if k == 27:
        print("ESC Mouse: " + str(mouseX) + " " + str(mouseY))
        break
    elif k == ord('a'):
        aimingPoint = [mouseX, mouseY]
        print("Aiming Point: " + str(mouseX) + " " + str(mouseY))
        userMessage = "Double click square top left point than press (1)"
        userImage = warpFire.copy()
        writeMessageToImage(userImage, userMessage)
    elif k == ord('1'):
        squareTopLeftPoint = [mouseX, mouseY]
        print("Square Top Left Point: " + str(mouseX) + " " + str(mouseY))
        userMessage = "Double click square top right point than press (2)"
        userImage = warpFire.copy()
        writeMessageToImage(userImage, userMessage)
    elif k == ord('2'):
        squareTopRightPoint = [mouseX, mouseY]
        print("Square Top Right Point: " + str(mouseX) + " " + str(mouseY))
        break

pixelDistance = calculatePixelDistance(squareTopLeftPoint, squareTopRightPoint)
print(pixelDistance)

distancePerPixel = 100 / pixelDistance
print(distancePerPixel)

distances = []

# Draw circles that are detected. 
if detectedCircles is not None: 
    # Convert the circle parameters a, b and r to integers. 
    # detectedCircles = np.uint16(np.around(detectedCircles)) 
    for pt in detectedCircles[0, :]: 
        bulletHole = [pt[0], pt[1]]
        pixelDistance = calculatePixelDistance(aimingPoint, bulletHole)
        mmDistance = pixelDistance * distancePerPixel
        print(mmDistance)
        distances.append(mmDistance)
        # Draw the circumference of the circle. 
        # Draw a small circle (of radius 1) to show the center. 
        #cv2.circle(imageFire, (a, b), 1, (0, 0, 255), 3) 

mean = st.mean(distances)
stdev = st.stdev(distances)

print("Mean(mm):" + str(mean))
print("Standard Deviation(mm):" + str(stdev))

# Displays subplotted images
# plt.show()
