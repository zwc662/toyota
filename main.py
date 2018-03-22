import numpy as np
import time
import scipy as sci
import mdp
from discretizer import discretizer
import os
import ast
from preprocess import preprocess_dict, preprocess_list
import scipy.sparse as sparse


<<<<<<< HEAD
=======

>>>>>>> d59a1548e32979cb1ae4dc01ccdb7bc7e1afaf43
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
        
<<<<<<< HEAD
        M = mdp.mdp()
=======
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
>>>>>>> d59a1548e32979cb1ae4dc01ccdb7bc7e1afaf43
        
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
<<<<<<< HEAD
	'''
	M.reward = np.random.random([len(M.S), ]).astype(float)
		
=======
>>>>>>> d59a1548e32979cb1ae4dc01ccdb7bc7e1afaf43
        start = time.time()
	policy = M.value_iteration()            
    	end = time.time()
	print("Value iteration time: %f" % (end - start))

<<<<<<< HEAD
        M.set_policy(policy)
	'''
	M.P = np.random.random((len(M.S), len(M.S)))
	M.P = sparse.csr_matrix(M.P)
	M.P = sparse.diags(1.0/M.P.sum(axis = 1).A.ravel()).dot(M.P)
	
	start = time.time()
	mu = []
	for f in range(50):
		mu.append(M.expected_value(reward = M.features[:, f]))
    	end = time.time()
	print("Policy iteration time: %f" % (end - start),)
	print("Expected features")
	print(mu)

	M.features = []
	mu = []
	for f in range(50):
		M.features.append(np.random.random([len(M.S)]))
		mu.append(np.random.random([1]))
	mu = np.reshape(mu, [50])
	M.features = np.array(M.features).T.reshape([len(M.S), 50])
	expert = M.features[-2]
	_, theta, _ = M.QP(expert, [mu])

=======

        M.set_policy(policy)
   	 

	start = time.time()
        mu = M.expected_features()
    	end = time.time()
	print("Policy iteration time: %f" % (end - start))

	print(mu)
>>>>>>> d59a1548e32979cb1ae4dc01ccdb7bc7e1afaf43
        exit()
    
        ##Use script to run PRISM, somehow doesn't work. Still working on it.
        os.system('/home/zekunzhou/workspace/toyota_project/prism-4.4.beta-src/src/demos/run /home/zekunzhou/workspace/toyota_project/')
        os.system('/home/zekunzhou/workspace/toyota_project/prism-4.4.beta-src/bin/prism ./grid_world.pm ./grid_world.pctl')


if __name__ == "__main__":
    run = run()
    run.main() 
    


