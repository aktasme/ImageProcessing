############################################################# 
# Bird Counting Algorithm                                   #
#                                                           #
# BLG 513E Image Processing HW#3                            #
#                                                           #
# Copyright (C)  Mehmet AKTAS 0504181576                    #
#                                                           #
#############################################################
import cv2
import math
import sys
import glob
import os
import numpy as np
from matplotlib import pyplot as plt

# Classifier Base Class
class ClassifierBase(object):
    # Load OpenCV SVM training data
    def load(self, filename):
        self.model.load(filename)

    # Save OpenCV SVM training data
    def save(self, filename):
        self.model.save(filename)

# SVM Class
class SVM(ClassifierBase):
    # SVM Constructor
    def __init__(self):
        # Create new SVM instance
        self.model = cv2.ml.SVM_create()

    def train(self, trainingData, labels):
        # Setting SVM algorithm parameters
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setKernel(cv2.ml.SVM_LINEAR)
        self.model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5))
        # Training process
        self.model.trainAuto(trainingData, cv2.ml.ROW_SAMPLE, labels)

    def predict(self, testData):
        # Prediction of new data
        output = self.model.predict(testData)
        return output
 
class ColorClassifier(object):
    # ColorClassifier Constructor
    # Creates new SVM instance and initialize attributes
    def __init__(self, colors, trainFile, testDirectory):
        self.colors = colors
        self.trainFile = trainFile
        self.testDirectory = testDirectory
        self.svm = SVM()
        self.trainingData = np.matrix([], dtype = np.float32)
        self.imageFeatures = np.array([], dtype = np.float32)
        self.labels = np.array([], dtype = np.int32)
        self.numberOfTrainingImages = 0
        self.histogramSizeInChannel = 16
        self.histogramSize = 3 * self.histogramSizeInChannel

    # Extract color histogram with given size (self.histogramSizeInChannel)
    def extractColorHistogramFromImage(self, image):
        channels = cv2.split(image)
        colors = ("b", "g", "r")
        featureList = []
        # Loop over the image channels (B, G, R)
        for (channel, color) in zip(channels, colors):
            # Calculate color histogram for each channel
            histogram = cv2.calcHist([channel], [0], None, [self.histogramSizeInChannel], [0, 256])
            # Concatenate histogram of BGR channels
            featureList.append(histogram)
        # Convert to np array and flatten
        feaureArray = np.array(featureList, dtype = np.float32)
        feaureArray = feaureArray.flatten()
        self.numberOfTrainingImages += 1
        return feaureArray

    # Read training image names from file and construct training data
    def prepareTrainingData(self):
        # Read given file and take image names
        with open(self.trainFile, 'r') as f:
            lines = [line.rstrip() for line in f]
        # Read all images and construct training data      
        for line in lines:
            # Argument[0] is image file name
            # Argument[1] is color of the image
            arguments = line.split()
            imageString = arguments[0]
            self.labels = np.append(self.labels, int(arguments[1]))
            image = cv2.imread(imageString)
            imageFeatures = self.extractColorHistogramFromImage(image)
            self.imageFeatures = np.append(self.imageFeatures, imageFeatures)          
        self.trainingData = self.imageFeatures.reshape(self.numberOfTrainingImages, self.histogramSize)

    # Train SVM from constructed training data
    def train(self):
        self.prepareTrainingData()
        # SVM training
        self.svm.train(self.trainingData, self.labels)
        # Save training data
        self.svm.save("svm.dat")

    # Find the image files in given directory and test each one
    # Calculate the accuracy
    def test(self):
        numberOfSuccessClassification = 0
        numberOfFailedClassification = 0
        for path, subdirs, files in os.walk(self.testDirectory):
            for file in files:
                if file.endswith(".jpg"):
                    testFile = os.path.join(path, file)
                    testImage = cv2.imread(testFile)
                    testFeatures = self.extractColorHistogramFromImage(testImage);
                    testData = testFeatures.reshape(1, self.histogramSize)
                    # SVM prediction
                    output = self.svm.predict(testData)
                    index = int(output[1])
                    color = colors[index]
                    # Compare the predicted color and groundtruth
                    if color in path:
                        result = "success"
                        numberOfSuccessClassification += 1
                    else:
                        result = "fail"
                        numberOfFailedClassification += 1
                    print(testFile + " " + color + " " + result)
        # Print results
        print("SuccessCount: " + str(numberOfSuccessClassification))
        print("FailedCount: " + str(numberOfFailedClassification))
        print("Accuracy: " + str(numberOfSuccessClassification/(numberOfSuccessClassification+numberOfFailedClassification)) )


print("Color classification from Mehmet AKTAS (0504181576)")
print("Python Version:" + sys.version)
print("Opencv Version:" + cv2.__version__)

# Colors of the images
colors = np.array(["Black", "Blue", "Yellow", "Green", "Orange", "Red", "Violet", "White"])
colorClassifier = ColorClassifier(colors, "TrainingSet/TrainingSet.txt", "TestSet")
colorClassifier.train()
colorClassifier.test()
