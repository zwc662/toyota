from mdp import mdp
from learn import learn
import numpy as np
import scipy.optimize as optimize
from cvxopt import matrix, solvers
from sklearn.preprocessing import normalize
import inspect
from scipy import sparse
from itertools import product
from multiprocessing import Pool
import sys
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


class apirl(mdp, object):
    def __init__(self):
        if sys.version_info[0] >= 3: 
            super().__init__()
        else:
            super(apirl, self).__init__()
        self.exp_mu = None
        self.theta = None
        pass
    
    def human_demo(self, paths = "./data/trans"):
        exp_mu = np.zeros(self.features[0].shape)
        mu_temp = exp_mu
        num_paths = 0

        file = open(str(paths), 'r')
        for line_str in file.readlines():
            line = line_str.split('\n')[0].split(' ')
            t = int(float(line[0]))
            if t == 0:
               exp_mu = exp_mu + mu_temp
               num_paths += 1
               mu_temp = self.features[self.S[-2]]
               t += 1
            s = int(float(line[1]))
            mu_temp = mu_temp + self.features[s] * (self.discount**t)
        
        exp_mu = exp_mu + mu_temp
        exp_mu = exp_mu/num_paths

        print("%d demonstrated paths in total" % num_paths)
        print("Expert expected features are:")
        print(exp_mu)
        return exp_mu

    def optimal_policy(self, theta):
        if theta is None:
            theta = self.theta
        theta = np.reshape(theta/np.linalg.norm(theta, ord = 2), (self.features.shape[-1], 1))
        self.reward = np.reshape(np.dot(self.features, theta), (len(self.S), ))
        self.policy = self.value_iteration()
        self.set_policy(self.policy)
        mu = self.expected_features_manual() 
        return mu, self.policy

    def random_demo(self):
        self.set_policy_random()
        mu = self.expected_features_manual()
        return mu
    
    def iteration(self, exp_mu = None, max_iter = None):
        if exp_mu is None:
            exp_mu = self.exp_mu

        if max_iter is None:
            max_iter = self.max_iter
    
        mus = list()

        print("Generated initial policy")
        theta = np.random.random((len(self.features[0])))
        theta = theta/np.linalg.norm(theta, ord = 2)
        mu, policy = self.optimal_policy(theta)
        print("Initial policy features:")
        print(mu)

        err = float('inf')
	err_ = 0
	
        itr = 0
        
        diff = np.linalg.norm(exp_mu - mu, ord = 2)
        opt = (diff, theta, self.policy, mu)
        
        print("APIRL iteration start:")
        while err > self.epsilon and itr <= max_iter:
            print("\n\n\nAPIRL iteration %d, error = %f" % (itr, err))
            if abs(err - err_) < self.epsilon:
                print("Stuck in local optimum. End iteration")
                break
            err_ = err

            itr += 1
            mus.append(mu)
            theta, err  = self.QP(exp_mu, mus) 

            print("Previous candidates error:")
            print(err)

            print("New candidate policy weight:")
            print(theta)

            mu, policy  = self.optimal_policy(theta)

            print("New candidate policy features:")
            print(mu)

            diff = np.linalg.norm(mu - exp_mu, ord = 2)
            if diff < opt[0]:
                opt = (diff, theta, self.policy, mu)
        	print("Update best policy")

        if err <= self.epsilon:
            print("\epsilon-close policy is found. APIRL finished")
            return theta, policy, mu 
        else:
            print("Can't find \espsilon-close policy. APIRL stop")
            return opt[1], opt[2], opt[3]

    def run(self, epsilon = 1e-5, max_iter = 30):
        if True:
            #real = raw_input("learn from 1. human 2. optimal policy 3. random policy, 4. exit")
            real = 1

            if real == 4:
                return
            elif real == 1:
                self.exp_mu = self.human_demo()
            elif real == 2:
                self.exp_mu = self.optimal_policy()
            elif real == 3:
                self.exp_mu = self.random_demo()
            else:
                return
            theta, policy, mu = self.iteration(max_iter = max_iter)

        print("weight")
        print(theta)
        print("features")
        print(mu)

        file = open("./data/policy", 'w')
        file.write(str(len(self.S)) + ":states:" + str(len(self.A)) + ":action:probability\n")
        for s in self.S:
            for a in self.A:
                file.write(str(s) + ':' + str(a) + ':' + str(policy[s][a]) + '\n') 
        file.close()
                
        
#if __name__ == "__main__":
#    AL = apprenticeship_learning()
#    AL.run()
