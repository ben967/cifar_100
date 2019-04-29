#################################################################################################################
# A script to train on the cifar 100 dataset, for coarse labels only
#################################################################################################################

# Imports
import numpy as np
import datasetFunctions as data
import tensorflow as tf
import tensorflow.contrib.layers as lays
import os

# Global variables
trainFile = 'data/TrainingData.npy'
testFile = 'data/TestingData.npy'
saveDir = 'models/cifar_coarse.ckpt'
numClasses = 20
imageSize = 32
numChannels = 3
numEpochs = 50000
summaryIterations = 1000
learningRate = 0.001
keepProb = 0.8

#################################################################################################################
# Main code begins here
#################################################################################################################

# First, load up the dataset
cifarData = data.dataset(trainFile, testFile)

# Placeholder for data input
x = tf.placeholder(tf.float32, [None, imageSize, imageSize, numChannels])
print(x.shape)

# Placeholder for truth labels input
y_ = tf.placeholder(tf.float32, [None, numClasses])

# Define helper functions
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# First convolution layer
conv1 = lays.conv2d(x, 16, [1, 1], stride=2, padding='same')
print(conv1.shape)

# Second convolution layer
conv2 = lays.conv2d(conv1, 32, [1, 1], stride=2, padding='same')
print(conv2.shape)

# Third convolution layer
conv3 = lays.conv2d(conv2, 32, [1, 1], stride=2, padding='same')
print(conv3.shape)

# Reshape to flatten out image into 1D tensor
net_flat = tf.reshape(conv3, [-1, 4*4*32])
print(net_flat.shape)

# Fully connected layer 1
net = lays.fully_connected(net_flat, num_outputs=512, activation_fn=tf.nn.tanh)
print(net.shape)

# Fully connected layer 2
net = lays.fully_connected(net, num_outputs=256, activation_fn=tf.nn.tanh)
print(net.shape)

# Apply dropout to reduce overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(net, keep_prob)

# Matmul layer
W_fc2 = weight_variable([256, numClasses])
b_fc2 = bias_variable([numClasses])
matmulResult = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
print(matmulResult.shape)

# Softmax layer
net = lays.softmax(matmulResult)
print(net.shape)

# Create session and initialise variables
sess = tf.Session()

# Train and evaluate the model
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(net), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(net,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

# Create the saver
saver = tf.train.Saver(tf.trainable_variables())

# Run the iterations
for i in range(numEpochs):

    # Get the data batch here
    trainDataBatch, trainLabelBatchCoarse, trainLabelBatchFine = cifarData.getTrainBatch(250)
    if i%summaryIterations == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={x:trainDataBatch, y_: trainLabelBatchCoarse, keep_prob: 1.0})

        testDataBatchCoarse, testLabelBatchCoarse, testLabelBatchCoarse = cifarData.getTestBatch(250)
        test_accuracy = accuracy.eval(session=sess, feed_dict={x:testDataBatchCoarse, y_: testLabelBatchCoarse, keep_prob: 1.0})

        print("Step: " + str(i) + ", Train accuracy: " + str(train_accuracy) + ", Test accuracy: " + str(test_accuracy))
    train_step.run(session=sess, feed_dict={x: trainDataBatch, y_: trainLabelBatchCoarse, keep_prob: keepProb})

print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={
    x:testDataBatch, y_: testLabelBatchCoarse, keep_prob: 1.0}))

# Save the variables to disk.
save_path = saver.save(sess, + saveDir)
print("Model saved in file: %s" % save_path)