import argparse
import matplotlib
matplotlib.use('Agg')
import cPickle 
import numpy as np
from pylab import *
from matplotlib import pyplot as plt

from lstmsolver.networks import LSTM, TanhLayer, Stacked
from lstmsolver.Trainers import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, help = "Dataset to be used")
parser.add_argument("--epochs", type=int, help = "Number of epochs")
args = parser.parse_args()

filename = args.filename
epochs = args.epochs

f = open("../datasets/"+filename+"_train.pkl", 'rb')
train_x, train_y = cPickle.load(f)
f.close()

f = open("../datasets/"+filename+"_test.pkl", "rb")
test_x, test_y = cPickle.load(f)
f.close()

print "Using 10 examples for validation"

val_x = train_x[len(train_x)-10:]
val_y = train_y[len(train_y)-10:]

train_x = train_x[:len(train_x)-10]
train_y = train_y[:len(train_y)-10]

print train_x.shape, train_y.shape, ": TRAINING SET SIZE"
print val_x.shape, val_y.shape, ": VALIDATION SET SIZE"
print test_x.shape, test_y.shape, ": TEST SET SIZE"

print "Dataset loaded successfully..."

lstm = LSTM(1, 25)
tanh = TanhLayer(25, 1)

rnn = Stacked([lstm, tanh])

trainer = Trainer(rnn, 1e-4, train_x, train_y, val_x, val_y, 1, len(train_x[0]), epochs)

print "Starting training..."
trainer.train()

f = open("../results/"+filename+".pkl", "wb")
cPickle.dump(trainer.net, f)
f.close()

print "Saving some images..."
image = test_x[3].reshape((len(test_x[3]), 1))
target = test_y[3].reshape((len(test_x[3]), 1))
output = trainer.predict(image)

fig = plt.figure()
subplot(131)
plt.plot(image)
subplot(132)
plt.plot(target)
subplot(133)
plt.plot(output)

fig.savefig("../results/"+filename+"_"+str(epochs)+".jpg")
print "Saved successfully ..."



