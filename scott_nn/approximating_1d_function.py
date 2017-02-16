from neuralnet import Network
from neuralnet import FastNetwork
import random
from matplotlib import pyplot as plt
import numpy as np

def chart(ANN,f,x0,x1,res):
	xs = np.linspace(x0,x1,res)
	y_f = map(f,xs)
	y_nn = [ANN.run_forward([x]) for x in xs]
	plt.plot(xs,y_f)
	plt.plot(xs,y_nn)
	plt.show()

def map_to_1D_func(f):
	ANN = FastNetwork(1,[6,6,1]) # can use either network here
	x0 = -1.
	x1 = 1.
	res = 100
	lastx = 0.

	runs = 100000
	for i in xrange(runs):
		if i%(runs/10) == 0:
			print i*1./runs
		if random.random()<0.9:
			x = lastx - 0.05+0.1*random.random()
			if x>1.: x = 1.
			if x<-1.: x = -1.
			lastx = x
		else:
			x = random.random()*(x1-x0)+x0
		if i < 3*runs/4:
			ANN.train([x],[f(x)],.7)
		else:
			ANN.train([x],[f(x)],0.05)

	chart(ANN,f,x0,x1,res)

f = lambda x:x**6
map_to_1D_func(f)
