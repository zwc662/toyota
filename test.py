import numpy as np
from pycuda.curandom import rand as curand
import scipy.sparse as sparse
import pycuda.gpuarray as gpuarray
import pycuda.driver as pycu
import pycuda.autoinit
from pycuda.reduction import ReductionKernel
import numba.cuda as cuda 
from time import time
import mdptoolbox
from multiprocessing import Pool

class run:

    def __init__(self):
        #15000 states
        self.S = range(15000)
        
        #5 actions
        self.A = range(5)
        
        #Assuming rewards for each state
        self.R = np.random.randint(2, size = (len(self.S)))
        
        #MDP transition function should be (A, S, S)
        self.T = list()
	for a in self.A:
	    self.T.append([])
	
	#DTMC transition function should be (S, S)
	self.P = np.zeros((len(self.S), len(self.S)))

        #Initialize a random policy (A, S) with distribution over each action
        self.policy = sparse.random(len(self.A), len(self.S), density = 0.5).todense()
        #policy distribution is not normalized, DTMC probability will be normalized later
        print("Policy finished")


    def learn(self):
        VL = mdptoolbox.mdp.ValueIteration(np.array(self.T), self.R, 0.99, 1e-5, 10000, initial_value = 0)
        VL.run()
        
        VL = mdptoolbox.mdp.ValueIteration(np.array([self.P]), self.R, 0.99, 1e-5, 10000, initial_value = 0)
        VL.run()

    def MDP_trans(self, a):
            #Generate random number in the range of [0, 1] for each T[a] matrix
            #Turn T[a] into sparse matrix
        self.T[a] = (sparse.random(len(self.S), len(self.S)))
        
            #Normalize each row of T[a]
        self.T[a] = sparse.diags(1.0/self.T[a].sum(axis = 1).A.ravel()).dot(self.T[a])  
        
        
    def DTMC_trans(self, a):
        #Choose an action a, repeat policy[a] from (S, 1) to (S, S), times T[a]
        self.P = self.T[a].dot(sparse.bsr_matrix(np.repeat(np.reshape(self.policy[a], [len(self.S), 1]), len(self.S), axis = 1)))
        #Normalize transition probability for DTMC
        
def trans(M):
    p = Pool(len(M.A))
    p.map(M.MDP_trans, M.A)
    print("Transition for action %d finished" % a)

    p.map(M.DTMC_trans, M.A)
    M.P = sparse.diags(1.0/M.P.sum(axis = 1).A.ravel()).dot(M.P)
    print("Transition for DTMC finished")

if __name__ == '__main__':
        test = run()
	trans(test)
	test.learn()
