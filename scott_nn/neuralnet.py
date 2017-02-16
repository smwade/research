import random
import math
from matplotlib import pyplot as plt
import numpy as np

class Neuron():
	def __init__(s,n):
		# n is number of inputs Func will get
		s.n = n
		s.wts = [random.random()*2.-1. for i in xrange(n)]
		s.addon = 0.
	def value(s,inputs):
		s.inp = inputs
		tot = sum(map(lambda x,y:x*y,inputs,s.wts))+s.addon
		s.val = 1/(1.+math.exp(-tot))
		return s.val
	def update(s,pull):
		mult = pull*s.val*(1-s.val)
		pulls = []
		for i in xrange(s.n):
			pulls.append(s.wts[i]*mult)
			s.wts[i] += s.inp[i]*mult
		s.addon += mult
		return pulls

class Layer():
	def __init__(s,n,m):
		s.n = n # number of nodes in layer
		s.m = m # number of inputs to each node (size of previous layer)
		s.nodes = [Neuron(m) for i in xrange(n)]
	def prop_forward(s,inputs):
		values = [node.value(inputs) for node in s.nodes]
		return values
	def update_backward(s,from_pulls):
		to_pulls = [0.]*s.m
		for node, from_pull in zip(s.nodes,from_pulls):
			for i, to_pull in enumerate(node.update(from_pull)):
				to_pulls[i] += to_pull
		return to_pulls

class Network():
	def __init__(s,num_inputs,layer_sizes):
		s.n = len(layer_sizes)
		s.layers = [Layer(layer_sizes[0],num_inputs)]
		for i in xrange(1,s.n):
			s.layers.append(Layer(layer_sizes[i],layer_sizes[i-1]))
	def run_forward(s,data):
		for layer in s.layers:
			data = layer.prop_forward(data)
		return data
	def run_backward(s,pulls):
		for layer in reversed(s.layers):
			pulls = layer.update_backward(pulls)
	def train(s,data,result,factor):
		output = s.run_forward(data)
		pulls = [factor*(r-o) for r,o in zip(result,output)]
		s.run_backward(pulls)
		return output

class FastNetwork():
	def __init__(s,num_inputs,layer_sizes):
		s.n = len(layer_sizes)
		s.vals = [None]*(len(layer_sizes)+1)
		s.layer_wts = [2*np.random.rand(layer_sizes[0],num_inputs)-1]
		s.layer_bias = [2*np.random.rand(layer_sizes[0])-1]
		for i in xrange(1,s.n):
			s.layer_wts.append(2*np.random.rand(layer_sizes[i],layer_sizes[i-1])-1)
			s.layer_bias.append(2*np.random.rand(layer_sizes[i])-1)
	def run_forward(s,data):
		s.vals[0] = data
		for i in xrange(s.n):
			data = np.dot(s.layer_wts[i],data) + s.layer_bias[i]
			data = 1./(1.+np.exp(-data))
			s.vals[i+1] = data
		return data
	def run_backward(s,pulls):
		for i in xrange(s.n-1,-1,-1):
			mults = pulls*s.vals[i+1]*(1-s.vals[i+1])
			pulls = np.dot(mults,s.layer_wts[i])
			s.layer_wts[i] += np.outer(mults,s.vals[i])
			s.layer_bias[i] += mults	
	def train(s,data,result,factor):
		output = s.run_forward(data)
		pulls = factor*(result-output)
		s.run_backward(pulls)
		return output
