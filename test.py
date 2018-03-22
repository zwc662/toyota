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

S = range(15000)
A = range(5)
R = np.random.randint(2, size = (len(S)))

T = list()
scale = np.random.randint(100, size = (len(S), len(S)))
for a in A:
    T.append(sparse.bsr_matrix(np.dot(scale, np.random.randint(2, size = (len(S), len(S))))))
    T[a] = sparse.diags(1.0/T[a].sum(axis = 1).A.ravel()).dot(T[a])  
print("Transition for MDP finished")

policy = sparse.bsr_matrix(np.random.random([len(S), len(A)]).astype(float))
print("Policy finished")

P = T[0].dot(sparse.bsr_matrix(np.repeat(np.reshape(policy.T[0], [len(S), 1]), len(S), axis = 1)))
for a in A[1:]:
    P += T[a].dot(sparse.bsr_matrix(np.repeat(np.reshape(policy.T[a], [len(S), 1]), len(S), axis = 1)))
P = sparse.diags(1.0/P.sum(axis = 1).A.ravel()).dot(P)
print("Transition for DTMC finished")

VL = mdptoolbox.mdp.ValueIteration(np.array(T), R, 0.99, 1e-5, 10000, initial_value = 0)
VL.run()

VL = mdptoolbox.mdp.ValueIteration(np.array([P]), R, 0.99, 1e-5, 10000, initial_value = 0)
VL.run()

