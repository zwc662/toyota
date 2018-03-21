import numpy as np
import time
import scipy as sci
from mdp import mdp
from discretizer import discretizer
import os
import ast
from preprocess import preprocess_dict, preprocess_list



class run(object):
    def __init__(maxsize):
	pass

#    def limit_memory(self):
#	maxsize = self.maxsize
#        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
#        resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))
    

    def main(self):
        #preprocess_dict('/home/zekunzhou/workspace/toyota_project/data/data.json')
        preprocess_list('/home/zekunzhou/workspace/toyota_project/data/data.json')
        
        M = mdp()
        '''
        #Simple MDP for testing
        M.build_from_config(4,2)
        M.set_starts([0])
        M.set_targets([2])
        M.set_unsafes([3])
        print(M.T)
        policy = np.ones([6, 2])
        policy[:, 0] = policy[:, 0] * 0.7
        policy[:, 1] = policy[:, 1] * 0.3
        print(policy)
        M.set_policy(policy)
        print(M.P)
        '''
        
        M.build_from_discretizer(num_actions = 5)
        M.preprocess_list("/home/zekunzhou/workspace/toyota_project/data/demo")
        M.set_transitions("/home/zekunzhou/workspace/toyota_project/data/transitions")
   
	''' 
        #Randomly generate a policy
        policy = np.random.randint(0, 5, len(M.S))
        #In each state there is a distribution of actions to take
        policy = np.random.rand(len(M.S), len(M.A))
        policy_ = np.sum(policy, axis = 1).reshape([len(M.S), 1])
        policy = policy / policy_
        #policy = policy + (1.0 - policy.sum(axis = 1)).reshape([len(M.S), 1]) * temp
        print(policy.sum(axis = 1))
        M.set_policy(policy)
        
     	M.output()
    	'''
        start = time.time()
	policy = M.value_iteration()            
    	end = time.time()
	print("Value iteration time: %f" % (end - start))


        M.set_policy(policy)
   	 

	start = time.time()
        mu = M.expected_features()
    	end = time.time()
	print("Policy iteration time: %f" % (end - start))

	print(mu)
        exit()
    
        ##Use script to run PRISM, somehow doesn't work. Still working on it.
        os.system('/home/zekunzhou/workspace/toyota_project/prism-4.4.beta-src/src/demos/run /home/zekunzhou/workspace/toyota_project/')
        os.system('/home/zekunzhou/workspace/toyota_project/prism-4.4.beta-src/bin/prism ./grid_world.pm ./grid_world.pctl')


if __name__ == "__main__":
    run = run()
    run.main() 
    


