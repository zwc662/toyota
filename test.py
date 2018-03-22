import numpy as np
from pycuda.curandom import rand as curand
import scipy.sparse as sparse
import pycuda.gpuarray as gpuarray
import pycuda.driver as pycu
import pycuda.autoinit
from pycuda.reduction import ReductionKernel
import numba.cuda as cuda 
from time import time

dot = ReductionKernel(dtype_out=np.float32, neutral="0",
                      reduce_expr="a+b", map_expr="x[i]*y[i]",
                      arguments="float *x, float *y")
n = 15000
x = curand((n), dtype=np.float32)
y = curand((n), dtype=np.float32)

x_cpu = gpuarray.to_gpu(np.random.random((n)))
y_cpu = gpuarray.to_gpu(np.random.random((n)))

st = time()
x_dot_y = dot(x_cpu, y_cpu).get()
gpu_time = (time() - st)
print "GPU: ", gpu_time

st = time()
x_dot_y_cpu = np.dot(x_cpu, y_cpu)
cpu_time = (time() - st)
print "CPU: ", cpu_time
print "speedup: ", cpu_time/gpu_time
