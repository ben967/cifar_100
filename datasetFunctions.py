# Imports
import numpy as np
import glob
import cv2
import random

# Global variables
imgSize = 32
numChannels = 3
coarseLabelDims = 20
fineLabelDims = 100

# Dataset class
class dataset:

    def __init__(self, trainFile, testFile):
        self.trainBatchIndex = 0
        self.testBatchIndex = 0
        self.trainFile = trainFile
        self.testFile = testFile
        self.trainDataFiles = np.load(self.trainFile)
        self.numTrainDataFiles = len(self.trainDataFiles)
        self.testDataFiles = np.load(self.testFile)
        self.numTestDataFiles = len(self.testDataFiles)

        # Print a bit of useful dataset info
        print("\nDataset Info")
        print("Training Data file: " + self.trainFile)
        print("Testing Data file: " + self.testFile)
        print("Number of training Data: " + str(self.numTrainDataFiles))
        print("Number of testing Data: " + str(self.numTestDataFiles))
        print("\n")



    # Get a training batch
    def getTrainBatch(self, numSamples):
        dataBatch = np.empty((0,imgSize,imgSize, numChannels))
        coarseLabelBatch = np.empty((0,coarseLabelDims))
        fineLabelBatch = np.empty((0,fineLabelDims))

        if (self.trainBatchIndex + numSamples) > self.numTrainDataFiles:
            self.trainBatchIndex = 0

        for x in range(self.trainBatchIndex, self.trainBatchIndex + numSamples):                
            data = self.trainDataFiles[x,:]
            coarseLabel = data[0]
            fineLabel = data[1]
            img = data[2:]
            img = np.dstack([img[2048:3072].reshape((32,32)), img[1024:2048].reshape((32,32)), img[0:1024].reshape((32,32))])
            img = np.interp(img,[0,255],[0,1]).astype(np.float32)
            img = cv2.resize(img, (imgSize, imgSize))
            img = img.reshape((1,imgSize,imgSize, numChannels))
            dataBatch = np.vstack([dataBatch, img])

            label = np.zeros((coarseLabelDims))
            label[coarseLabel] = 1
            label = label.reshape((1, coarseLabelDims))
            coarseLabelBatch = np.vstack([coarseLabelBatch, label])

            label = np.zeros((fineLabelDims))
            label[fineLabel] = 1
            label = label.reshape((1, fineLabelDims))
            fineLabelBatch = np.vstack([fineLabelBatch, label])

        self.trainBatchIndex += numSamples
        return dataBatch, coarseLabelBatch, fineLabelBatch
        


    # Get a testing batch
    def getTestBatch(self, numSamples):
        dataBatch = np.empty((0,imgSize,imgSize, numChannels))
        coarseLabelBatch = np.empty((0,coarseLabelDims))
        fineLabelBatch = np.empty((0,fineLabelDims))

        if (self.testBatchIndex + numSamples) > self.numTestDataFiles:
            self.testBatchIndex = 0

        for x in range(self.testBatchIndex, self.testBatchIndex + numSamples):                
            data = self.testDataFiles[x,:]
            coarseLabel = data[0]
            fineLabel = data[1]
            img = data[2:]
            img = np.dstack([img[2048:3072].reshape((32,32)), img[1024:2048].reshape((32,32)), img[0:1024].reshape((32,32))])
            img = np.interp(img,[0,255],[0,1]).astype(np.float32)
            img = cv2.resize(img, (imgSize, imgSize))
            img = img.reshape((1,imgSize,imgSize, numChannels))
            dataBatch = np.vstack([dataBatch, img])

            label = np.zeros((coarseLabelDims))
            label[coarseLabel] = 1
            label = label.reshape((1, coarseLabelDims))
            coarseLabelBatch = np.vstack([coarseLabelBatch, label])

            label = np.zeros((fineLabelDims))
            label[fineLabel] = 1
            label = label.reshape((1, fineLabelDims))
            fineLabelBatch = np.vstack([fineLabelBatch, label])

        self.trainBatchIndex += numSamples
        return dataBatch, coarseLabelBatch, fineLabelBatch