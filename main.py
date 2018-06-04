import numpy as np
import time
import scipy as sci
import mdp
import apirl
from discretizer import discretizer
import os
import ast
from preprocess import preprocess_dict, preprocess_list
import scipy.sparse as sparse



class toyota(grids, object):
    def __init__(self):
        if sys.version_info[0] >= 3:  
            super().__init__()
        else:
            super(cartpole, self).__init__()
        ##[front dist, rear dist, left, right, front speed, rear speed, left speed, right speed, lane pos]
	'''
        self.threshes = [
                         [5],#front dist
                         [5],#rear dist
                         [1],#left
                         [1],#right
                         [5],#front speed
                         [-30],#rear speed
                         [-30],#left speed
                         [-30],#right speed
                         [-42.0]#lane pos
                         ] ##list of threshes for each dimension
        

        # Thresholds for each interval
        self.threshes = [
                         [5, 15, 25],  # front dist
                         [5, 15, 25],  # rear dist
                         [1],  # left
                         [1],  # right
                         [-5, 5],  # front speed
                         [-5, 5],  # rear speed
                         [-5, 5],  # left speed
                         [-5, 5],  # right speed
                         [-42.0, -40.5],  # lane pos
                         ]  # list of threshes for each dimension
        '''
        self.threshes = [
                     [-41],  # car1 lane position
                     [0, 20, 40],  # car1 velocity
                
                     [-41],  # car2 lane
                     [-30, -10, 10, 30],  # car2 distance
                     [-10, 0, 10],  # car2 speed
                         
                     [-41],  # car3 lane
                     [-30, -10, 10, 30],  # car3 distance
                     [-10, 0, 10],  # car3 speed
                     ]  # list of threshes for each dimension
        self.grids = []

#    def limit_memory(self):
#	maxsize = self.maxsize
#        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
#        resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))
    

    def main(self):
        #preprocess_dict('/home/zekunzhou/workspace/toyota_project/data/data.json')
        #preprocess_list('/home/zekunzhou/workspace/toyota_project/data/data.json')
        
        #M = mdp.mdp()
    
	M = apirl.apirl()
        
        M.build_from_discretizer(num_actions = 2 * 3)
        M.preprocess_list()
        M.set_transitions()
        
        M.run(max_iter = 50)

        M.build_from_config(15000, 5)
        M.set_transitions_random()
        M.set_policy_random()

        M.reward = np.random.random([len(M.S), ]).astype(float)
        
        start = time.time()
        _, w, _ = M.QP(np.random.random((10)), np.random.random((10, 10)))
        end = time.time()
        print("QP time: %f" % (end - start))

        start = time.time()
        policy = M.value_iteration()
        end = time.time()
        print("Value iteration time: %f" % (end - start))

        M.set_policy(policy)

        start = time.time()
        v = M.expected_value()
        #mu = M.LP_features()
        end = time.time()
        print("Policy iteration time: %f" % (end - start))
        print(v[-2])

        '''
        start = time.time()
        v = M.expected_value_manual()
        #mu = M.LP_features()
            end = time.time()
        print("Policy iteration time: %f" % (end - start))
        print(v[-2])
        '''

        start = time.time()
        v = M.LP_value_()
        end = time.time()
        print("LP time: %f" % (end - start))
        print(v)
        print(v[-2])
        #mu = M.LP_features()

    
        exit()
        ##Use script to run PRISM, somehow doesn't work. Still working on it.
        os.system('/home/zekunzhou/workspace/toyota_project/prism-4.4.beta-src/src/demos/run /home/zekunzhou/workspace/toyota_project/')
        os.system('/home/zekunzhou/workspace/toyota_project/prism-4.4.beta-src/bin/prism ./grid_world.pm ./grid_world.pctl')


if __name__ == "__main__":
    run = run()
    run.main() 
    


