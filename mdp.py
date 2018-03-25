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

@vectorize(["float64(float64, float64)"], target='cuda')
def VectorDot(a, b):
	return a * b


dot = ReductionKernel(dtype_out=np.float64, neutral="0",
                      reduce_expr="a+b", map_expr="x[i]*y[i]",
                      arguments="float *x, float *y")

class mdp:

    def __init__(self):
        self.S = list()
        # Always add two extra states at the end
        # S[-1] is the single unsafe terminal and S[-2] is the sngle initial terminal
        # All unsafe states have probability 1 to reach the unsafe terminal
        # From initial terminal there is a distribution of transiting to the
        # initial states
        self.A = list()
        # List of actions
        self.T = None
        # A list of numpy transition matrices for each actions
        ## [T(_, a_0, _), T(_, a_1, _), ...]
        self.P = None
        # The transition probability of DTMC given a policy
        self.policy = None
        # Policy is a |S|x|A| matrix of distributions actions for each states
        self.targets = list()
        # A list of absorbing target state to be reached, may be empty
        self.starts = list()
        # A list of initial states
        self.unsafes = list()
        # A list of unsafe states
	self.features = None
	self.reward = None
	# Features of all states
        self.discretizer = discretizer()
        # Include the discretizer as a member

    def build_from_config(self, num_states, num_actions):
        # Only have the number of states and actions, build the MDP
        self.S = range(num_states + 2)  # Add two external source
        self.A = range(num_actions)
        self.T = []
        for a in range(len(self.A)):
            self.T.append(np.zeros([len(self.S), len(self.S)], dtype=np.float64))
        # Init transition matrices to be all zero
	self.set_features()
        print("construction complete")

    def build_from_discretizer(self, discretizer=None, num_actions=None):
        # Given a discretizer, build the MDP
        if discretizer is not None:
            self.discretizer = discretizer
        # print(self.discretizer.grids)
        self.S = range(np.array(self.discretizer.grids).prod() + 2)
        # Add two external source
        print(len(self.S))
        self.A = range(num_actions)
        print(len(self.A))
        self.T = []
        for a in range(len(self.A)):
            self.T.append(np.zeros([len(self.S), len(self.S)], dtype=np.float64))
        # Init transition matricies to be all zero
	self.set_features()
        print("construction complete")

    def set_starts(self, starts):
        # Set uniform distribution of starting from each initial states
        self.starts = starts
        for a in range(len(self.A)):
            self.T[a][:, self.S[-2]] = self.T[a][:, self.S[-2]] * 0.0
            self.T[a][self.S[-2]] = self.T[a][self.S[-2]] * 0.0

            self.T[a][:, self.starts] = self.T[a][:, self.starts] * 0.0
            self.T[a][self.S[-2], self.starts] = 1.0/len(self.starts)

    def set_targets(self, targets):
        # Set target states to be absorbing
        self.targets = targets
        for a in range(len(self.A)):
            self.T[a][self.targets] = self.T[a][self.targets] * 0.0
            self.T[a][self.targets, self.targets] = 1.0

    def set_unsafes(self, unsafes):
        # Set all probabilities of transitioning from unsafe states to unsafe
        # terminal to be 1
        self.unsafes = unsafes

        for a in range(len(self.A)):
            self.T[a][:, self.S[-1]] = self.T[a][:, self.S[-1]] * 0.0
            self.T[a][self.S[-1]] = self.T[a][self.S[-1]] * 0.0
            self.T[a][self.S[-1], self.S[-1]] = 1.0

            self.T[a][self.unsafes] = self.T[a][self.unsafes] * 0.0
            self.T[a][self.unsafes, self.S[-1]] = 1.0

    def set_features(self):
	self.features = (np.random.random(size = [len(self.S),50]).astype(np.float64))

    def observation_to_state(self, observation):
        # Translate observation to coordinate by calling the discretizer
        # Then translate the coordinate to state
        coord = self.discretizer.observation_to_coord(observation)
        state = coord[0]
        for i in range(1, len(coord)):
            state_temp = coord[i]
            for j in range(0, i):
                state_temp *= self.discretizer.grids[j]
            state += state_temp
        return state

    def preprocess_list(self, data):
        # Preprocess the data set file which contains trajectories
        # Each trajectory is a list in which each element in a list is a list of time step, dict of observations and ...
        # This method translates the observations to states
        tuples = []
        file_i = open(data, 'r')
        print("read list file")
        for line_str in file_i.readlines():
            line = ast.literal_eval(line_str)
            observation = line[1]
            state = self.observation_to_state(observation)
            action = line[2]
            observation_ = line[3]
            state_ = self.observation_to_state(observation_)
            tuples.append(
                (str(state) +
                 ' ' +
                 str(action) +
                    ' ' +
                    str(state_) +
                    '\n'))
        file_i.close()

        file_o = open('./data/transitions', 'w')
        for line_str in tuples:
            file_o.write(line_str)
        file_o.close()

    def set_transitions(self, transitions):
        # Count the times of transitioning from one state to another
        # Calculate the probability
        # Give value to self.T
        file = open(str(transitions), 'r')
        for line_str in file.readlines():
            line = line_str.split('\n')[0].split(' ')
            a = int(float(line[1]))
            s = int(float(line[0]))
            s_ = int(float(line[2]))
            self.T[a][s, s_] += 1
        file.close()
        for a in range(len(self.A)):
            for s in range(len(self.S)):
                tot = np.sum(self.T[a][s])
                if tot == 0.0:
                    self.T[a][s,s] = 1.0
	    self.T[a] = sparse.bsr_matrix(self.T[a])
	    self.T[a] = sparse.diags(1.0/self.T[a].sum(axis = 1).A.ravel()).dot(self.T[a])


    def set_transitions_random(self):
        # Count the times of transitioning from one state to another
        # Calculate the probability
        # Give value to self.T
	self.T = list()
        for a in range(len(self.A)):
	    self.T.append(sparse.random(len(self.S), len(self.S), density = 0.1).todense())
            for s in range(len(self.S)):
                tot = np.sum(self.T[a][s])
                if tot == 0.0:
                    self.T[a][s,s] = 1.0
	    self.T[a] = sparse.bsr_matrix(self.T[a])
	    self.T[a] = sparse.diags(1.0/self.T[a].sum(axis = 1).A.ravel()).dot(self.T[a])

    def set_policy_random(self):
        self.policy = np.random.random((len(self.S), len(self.A))) 
        self.P = sparse.bsr_matrix(np.zeros([len(self.S), len(self.S)], dtype=np.float64))
        for a in range(len(self.A)):
            self.P += self.T[a].dot(sparse.bsr_matrix(np.repeat(np.reshape(self.policy.T[a], [len(self.S), 1]), len(self.S), axis = 1 )))
	self.P = sparse.diags(1.0/self.P.sum(axis = 1).A.ravel()).dot(self.P)
        
        print("DTMC transition constructed")


    def set_policy(self, policy):
	if isinstance(policy, sparse.csr_matrix) or isinstance(policy, sparse.csc_matrix):
        	self.policy = policy.todense()
	else:
		self.policy = policy

	assert self.policy.shape == (len(self.S), len(self.A))
        self.P = sparse.bsr_matrix(np.zeros([len(self.S), len(self.S)], dtype=np.float64))
        for a in range(len(self.A)):
            self.P += self.T[a].dot(sparse.bsr_matrix(np.repeat(np.reshape(self.policy.T[a], [len(self.S), 1]), len(self.S), axis = 1 )))
	self.P = sparse.diags(1.0/self.P.sum(axis = 1).A.ravel()).dot(self.P)

        print("DTMC transition constructed")

    def output(self):
        # Output files for PRISM
        os.system('rm ./state_space')
        os.system('touch ./state_space')
        file = open('./state_space', 'w')
        file.write('states\n' + str(len(self.S)) + '\n')
        file.write('actions\n' + str(len(self.A)) + '\n')
        file.close()

        os.system('rm ./unsafe')
        os.system('touch ./unsafe')
        file = open('unsafe', 'w')
        for i in range(len(self.unsafes)):
            file.write(str(self.unsafes[i]) + ':\n')
        file.close()

        os.system('rm ./optimal_policy')
        os.system('touch ./optimal_policy')
        file = open('./optimal_policy', 'w')

        def writer(tup, f):
            file.write(str(self.S[i]) +
                       ' ' +
                       str(self.S[j]) +
                       ' ' +
                       str(self.P[self.S[i], self.S[j]]) +
                       '\n')

        print("filtering self loops")
        fil_S = filter(lambda x: self.P[self.S[x], self.S[
                       x]] < 0.99999, range(len(self.S)))
        print("making transitions")
        transitions = product(fil_S, repeat=2)
        transitions = filter(
            lambda tup: self.P[
                self.S[
                    tup[0]], self.S[
                    tup[1]]] > 1e-5, transitions)
        p = Pool(4)
        print("mapping writer on transitions")
        p.map(lambda tup: writer(tup, file), transitions)
        # for i in range(len(self.S)):
        #	if self.P[self.S[i], self.S[i]] > 1.0 -  1E-5:
        #		continue
        #	for j in range(len(self.S)):
        #		if self.P[self.S[i], self.S[j]] > 1E-5:
        #			file.write(str(self.S[i]) + ' ' + str(self.S[j]) + ' ' + str(self.P[self.S[i], self.S[j]]) + '\n')
        #			print(str(self.S[i]) + ' ' + str(self.S[j]) + ' ' + str(self.P[self.S[i], self.S[j]]))

        file.close()

		
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
