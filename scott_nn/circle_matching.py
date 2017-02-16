from neuralnet import Network
from neuralnet import FastNetwork
import random
from matplotlib import pyplot as plt

def dist(x,y,a,b):
	return ((x-a)**2+(y-b)**2)**0.5

ANN = FastNetwork(2,[8,4,1]) # can use either network here

def train():
	runs = 10000
	for i in xrange(runs):
		if i%(runs/10)==0:
			print i*1./runs

		x = 6*random.random()-3
		y = 6*random.random()-3

		if dist(x,y,-1.5,1.5) < .8 or dist(x,y,1.5,-1.5) < .8:
			ANN.train([x,y],[1.],0.5)
		else:
			ANN.train([x,y],[0.],0.5)

def plot():
	xs = []
	ys = []
	zs = []

	for i in xrange(5000):
		x = 2*random.random()-1.
		y = 2*random.random()-1.

		x = 6*random.random()-3
		y = 6*random.random()-3

		output = ANN.run_forward([x,y])[0]
		xs.append(x)
		ys.append(y)
		zs.append(output)

#	for i,layer in enumerate(ANN.layers):
#		print "Layer %d:" % i
#		for j,node in enumerate(layer.nodes):
#			print "   Node %d:" % j
#			print "     ", node.wts, node.addon

	circle1=plt.Circle((-1.5,1.5),.8,lw=5.,color='k',fill=False)
	circle2=plt.Circle((1.5,-1.5),.8,lw=5.,color='k',fill=False)

	plt.gca().add_artist(circle1)
	plt.gca().add_artist(circle2)
	plt.scatter(xs,ys,c=zs)

train()
plot()
plt.title('After 10,000 training points')
plt.show()

train()
plot()
plt.title('After 20,000 training points')
plt.show()

train()
plot()
plt.title('After 30,000 training points')
plt.show()

train()
plot()
plt.title('After 40,000 training points')
plt.show()
