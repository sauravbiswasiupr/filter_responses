#!/usr/bin/python 
'''Takes the blurring value as a parameter and constructs the dataset of sines 
such that frequency values are sampled from the range (1, 10) , w = 2*pi*f
and convolves them with a low pass filter that is gaussian'''

import argparse 
import numpy as np
import cPickle 
from scipy.ndimage.filters import gaussian_filter
import argparse
from pylab import *
import random 

def generateSines(fLow, fHigh, sigma, size):
  inps = []; targs = []
  x = np.linspace(-2*np.pi, 2*np.pi)
  for i in range(size):
    f = random.randint(fLow, fHigh)
    y = np.sin(2*np.pi*f*x)
    targ = gaussian_filter(y, sigma)
    inps.append(y)
    targs.append(targ)
  return np.array(inps), np.array(targs)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sigma", type=float, help = "the blur value for the gaussian filter")
parser.add_argument("--trainSize", type = int, help = "the size of training set")
parser.add_argument("--testSize", type = int, help = "the size of theb test set")
parser.add_argument("--freqlow", type =int, help = "the lower frequency value")
parser.add_argument("--freqhigh", type =int, help = "the higher frequency value")
parser.add_argument("--filename", type=str, help = "the filename to save the data as")

args = parser.parse_args()
sigma = args.sigma
trainsize = args.trainSize
testSize = args.testSize
filename = args.filename

fLow = args.freqlow
fHigh = args.freqhigh

train_x, train_y = generateSines(fLow, fHigh, sigma, trainsize)
test_x, test_y = generateSines(fLow, fHigh, sigma, testSize)

train = (train_x, train_y)
test  = (test_x, test_y)

print train_x.shape, train_y.shape 
print test_x.shape, test_y.shape 

f = open("../datasets/"+filename+"_train.pkl", "wb")
cPickle.dump(train, f)
f.close()

f = open("../datasets/"+filename+"_test.pkl", "wb")
cPickle.dump(test, f)
f.close()

print "Files saved successfully to disk..."
