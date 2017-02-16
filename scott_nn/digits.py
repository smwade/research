import cPickle, gzip
import numpy as np
from matplotlib import pyplot as plt
from neuralnet import Network
from neuralnet import FastNetwork

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

def maxind(l):
	ind = 0
	maxval = l[0]
	for i in xrange(1,len(l)):
		if l[i]>maxval:
			maxval = l[i]
			ind = i
	return ind

numdigits = 10
using = set(range(numdigits))
ANN = FastNetwork(784,[50,50,numdigits]) # can use either network here
correct = dict()
for i in xrange(numdigits):
	l = [0.]*numdigits
	l[i] = 1.
	correct[i] = np.array(l)

lim = 1000
history = []
count = 0
for j in xrange(1000):
	for i in xrange(len(train_set[0])):
		image, truth = train_set[0][i], train_set[1][i]
		if truth in using:
			val = ANN.train(image,correct[truth],0.5)
			guess = maxind(val)

			if guess == truth:
				history.append(1)
			else:
				history.append(0)
			count += 1

			alltime_ac = sum(history)*1./count
			latest_ac = sum(history[-lim:])*1./min(lim,count)
			print "%d-%d - Was %d, NN guessed %d - Accuracy for last %d: %.3f - All time: %.3f" % (j, count, truth, guess, lim, latest_ac, alltime_ac)
			der_fact = (1.-latest_ac) / 5.
