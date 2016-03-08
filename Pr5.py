import theanets
import scipy
import math
import numpy as np
import numpy.random as rnd
import logging
import sys
import collections
import theautil


# maximum number of iterations before we bail
mupdates = 1000

# setup logging
logging.basicConfig(stream = sys.stderr, level=logging.INFO)

data = np.loadtxt("yeast.csv", delimiter=",")
inputs  = data[0:,0:8].astype(np.float32)
outputs = data[0:,8:9].astype(np.int32)

# now lets shuffle
# If we're going to select a validation set we probably want to shuffle
def joint_shuffle(arr1,arr2):
    assert len(arr1) == len(arr2)
    indices = np.arange(len(arr1))
    np.random.shuffle(indices)
    arr1[0:len(arr1)] = arr1[indices]
    arr2[0:len(arr2)] = arr2[indices]

# our data and labels are shuffled together
joint_shuffle(inputs,outputs)


def split_validation(percent, data, labels):
    ''' 
    split_validation splits a dataset of data and labels into
    2 partitions at the percent mark
    percent should be an int between 1 and 99
    '''
    s = int(percent * len(data) / 100)
    tdata = data[0:s]
    vdata = data[s:]
    tlabels = labels[0:s]
    vlabels = labels[s:]
    return ((tdata,tlabels),(vdata,vlabels))


# make a validation set from the train set
train, valid = split_validation(90, inputs,outputs)


def linit(x):
    return x.reshape((len(x),))

train = (train[0],linit(train[1]))
valid = (valid[0],linit(valid[1]))
#test  = (test[0] ,linit(test[1]))


cnet = theanets.Classifier([8,2,2])
cnet.train(train,valid, algo='layerwise', patience=1, max_updates=mupdates)
cnet.train(train,valid, algo='rprop', patience=1, max_updates=mupdates)


print "%s / %s " % (sum(cnet.classify(inputs) == outputs),len(outputs))
classify = cnet.classify(valid[0])
print "%s / %s " % (sum(classify == valid[1]),len(valid[1]))
print collections.Counter(classify)
print theautil.classifications(classify,valid[1])

