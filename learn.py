from mdp import mdp
import numpy as np
import scipy.optimize as optimize
from cvxopt import matrix, solvers
from sklearn.preprocessing import normalize
import inspect
from scipy import sparse
from itertools import product
from multiprocessing import Pool
import os
import ast
import time
import mdptoolbox
from discretizer import discretizer
from timeit import default_timer as timer
from numba import vectorize
from pycuda.curandom import rand as curand
import pycuda.gpuarray as gpuarray
import pycuda.driver as pycu
import pycuda.autoinit
from pycuda.reduction import ReductionKernel
import numba.cuda as cuda 

from discretizer import discretizer
import mdptoolbox 


class learn(mdp):
    def __init__(self):
        super().__init__()
        
        
    def expected_features_manual(self, discount = 0.99, epsilon = 1e-5, max_iter = 10000):
	itr = 0
	mu_temp = self.features
	diff = float('inf')
	assert self.P.shape[1] == features.shape[0]
	assert self.P.shape[0] == features.shape[0]
	while diff > epsilon:
		itr += 1	
		print("Iteration %d, difference is %f" % (itr, diff))
		mu = mu_temp
		mu_temp = self.features + discount * (self.P.dot(mu))
                diff = (abs((mu_temp - mu).max()) + abs((mu_temp - mu).min()))/2
			
	return mu[len(self.S)-2]/discount
	
    def expected_features(self, discount = 0.99, epsilon = 1e-5, max_iter = 10000):
	mu = []
	for f in range(self.features.shape[-1]):
		V = self.features[:, f].reshape(len(self.S))
		VL = mdptoolbox.mdp.ValueIteration(np.array([self.P]), V, discount, epsilon, max_iter, initial_value = 0)
		itr = 0
		VL.run()
		mu.append(VL.V[-2])
	return mu

    def expected_value_manual(self, discount = 0.99, epsilon = 1e-5, max_iter = 10000):
        itr = 0
        v_temp = self.reward
	if type(self.P) is not type(self.T[0]):
		self.P = type(self.T[0])(self.P)
        diff = float('inf')
        while diff > epsilon:
            itr += 1
	    print("Iteration %d, difference is %f" % (itr, diff))
	    v = v_temp
	    v_temp = self.reward + discount * (self.P.dot(v))
            diff = (abs((v_temp - v).max()) + abs((v_temp - v).min()))/2
        return v
			

    def expected_value(self, discount = 0.99, epsilon = 1e-5, max_iter = 10000):
	if type(self.P) is not type(self.T[0]):
		self.P = type(self.T[0])(self.P)
	VL = mdptoolbox.mdp.ValueIteration(np.array([self.P]), self.reward, discount, epsilon, max_iter, initial_value = 0)
	print("Calculating expected value")
	VL.run()
	return VL.V

    def value_iteration(self, discount = 0.99, epsilon = 1e-5, max_iter = 10000):
	#M = mdptoolbox.mdp.MDP(np.array(self.T), reward, discount, epsilon, max_iter)
	VL = mdptoolbox.mdp.ValueIteration(np.array(self.T), self.reward, discount, epsilon, max_iter, initial_value = 0)
	VL.run()
	policy = np.zeros([len(self.S), len(self.A)]).astype(float)
 	for s in range(len(VL.policy)):
		policy[s, VL.policy[s]] = 1.0
	policy = sparse.csc_matrix(policy)
	return policy
		
	
    def expected_value_gpu(self, reward = None, discount = 0.99, epsilon = 1e-5, max_iter = 10000):
	if reward is None:
		reward = self.reward
	itr = 0
	value = reward
	diff = float('inf')
	self.P = self.P.todense()
	while diff > epsilon:
		itr += 1	
		value_temp = value.copy()
		#value = self.reward + discount * self.P.dot(value)
		for s in self.S:
			value[s] = reward[s] + discount * gpuarray.dot(gpuarray.to_gpu(self.P[s]), gpuarray.to_gpu(value)).get()
			
		var = (value[len(self.S) - 2] - value_temp[len(self.S) - 2])
		diff = abs(var.max()) +  abs(var.min())
		print("Iteration %d, difference is %f > error bound? %d" % (itr, diff, diff > epsilon))
	return value[len(self.S)-2]

    def QP(self, expert, features, epsilon = 1e-5):
	assert expert.shape[-1] == np.array(features).shape[-1]
	G_i = []
	h_i = []
	for k in range(len(expert)):
		G_i.append([0])	
	G_i.append([-1])
	h_i = [0]
	c = matrix(np.eye(len(expert) + 1)[-1] * -1)
	for j in range(len(features)):
		for k in range(len(expert)):
			G_i[k].append( - expert[k] + features[j][k])	
		G_i[len(expert)].append(1)
		h_i.append(0)
	for k in range(len(expert)):
		G_i[k] = G_i[k] + [0.0] * (k + 1) + [-1.0] + [0.0] * (len(expert) + 1 - k - 1)
	G_i[len(expert)] = G_i[len(expert)] + [0.0] * (1 + len(expert)) + [0.0]
	h_i = h_i + [1] + (1 + len(expert)) * [0.0]
	G = matrix(G_i)
	h = matrix(h_i)
	dims = {'l': 1 + len(features), 'q': [len(expert) + 1, 1], 's': []}
	start = time.time()
	sol = solvers.conelp(c, G, h, dims)
	end = time.time()
	print("QP operation time = " + str(end - start))
	print(sol.keys())
	print sol['status']
	solution = np.array(sol['x'])
	if solution is not None:
		solution=solution.reshape(len(expert) + 1)
		w = solution[:-1]
		t = solution[-1]
	else:
		w = None
		t = None
	
	return 0, w, t
	
    def LP_value(self, epsilon = 1e-5, discount = 0.5):
	if not isinstance(self.P, sparse.csr_matrix):
		self.P = sparse.csr_matrix(self.P)
	start = time.time()
	c = np.ones((len(self.S)))
    	A_ub = discount * self.P.transpose() - sparse.eye(len(self.S))
	assert A_ub.shape == (len(self.S), len(self.S))
	b_ub = -1 * self.reward
	sol = optimize.linprog(c = c, A_ub = A_ub.todense(), b_ub = b_ub, method = 'simplex')
	end = time.time()
	print('Solving one expected value via sparse LP, time = %f' % (end - start))
	return np.reshape(np.array(sol['x']), (len(self.S)))
	

    def LP_value_(self, epsilon = 1e-5, discount = 0.5):
    	self.P = self.P.todense()
	assert self.P.shape == (len(self.S), len(self.S))
    	start = time.time()
    	c = np.ones((len(self.S))).tolist()
    	G = (discount * self.P.T - np.eye(len(self.S))).tolist()
    	h = (-1 * self.reward).tolist()
    	sol = solvers.lp(matrix(c), matrix(G), matrix(h))
    	end = time.time()
    	print('Solving one expected value via LP, time = ' + str(end - start))
	return np.reshape(np.array(sol['x']), (len(self.S)))

    def LP_features(self, epsilon = 1e-5, discount = 0.5):
    	self.P = self.P.todense()
	assert self.P.shape == (len(self.S), len(self.S))
    	mu = []
    	for f in range(len(self.features[0])):
    		start = time.time()
    		c = np.ones((len(self.S))).tolist()
    		G = (discount * self.P.T - np.eye(len(self.S))).tolist()
    		h = (-1 * self.features[:, f]).tolist()
		print("Start solving feature %d..." % f)
    		sol = solvers.lp(matrix(c), matrix(G), matrix(h))
    		mu.append(np.array(sol['x']).reshape((len(self.S))))
		print("Finished solving feature %d..." % f)
    		end = time.time()
    		print('Solving one expected feature via LP, time = ' + str(end - start))
    	return mu
