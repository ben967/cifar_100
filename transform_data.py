#################################################################################################################
# A script to alter the cifar 100 dataset, to remove need for pandas
#################################################################################################################

# Imports
import numpy as np
import pickle

# Global variables
trainFile = 'data/train'
testFile = 'data/test'
metaFile = 'data/meta'

#################################################################################################################
# Main code begins here
#################################################################################################################

# Unpickle function from cifar 100 website
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

coarseLabels = []
fineLabels = []
data = []

trainData = unpickle(trainFile)
print(trainData.keys())
for key,value in trainData.items():
    if b'coarse_labels' in key:
        coarseLabels = value
    elif b'fine_labels' in key:
        fineLabels = value
    elif b'data' in key:
        data = value

coarseLabels = np.asarray(coarseLabels).reshape((50000,-1)).astype(np.uint8)
fineLabels = np.asarray(fineLabels).reshape((50000,-1)).astype(np.uint8)
data = np.asarray(data).reshape((50000,-1)).astype(np.uint8)

trainingData = np.hstack([coarseLabels, fineLabels, data])
print(trainingData.shape)
#np.savetxt('trainingData.csv', trainingData, delimiter=',')
np.save('TrainingData.npy', trainingData)
