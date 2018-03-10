import numpy as np
import scipy as sci
from mdp import mdp
from discretizer import discretizer
import os
import ast
from preprocess import preprocess_dict, preprocess_list

if __name__ == "__main__":
    print("hehe")
    preprocess_dict('/home/zekunzhou/workspace/toyota_project/data/data.json')
    preprocess_list('/home/zekunzhou/workspace/toyota_project/data/data.json')
    
    M = mdp()
    '''
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
    policy = np.random.randint(0, 5, len(M.S))
    policy = np.random.rand(len(M.S), len(M.A))
    policy_ = np.linalg.norm(policy, axis = 1).reshape([len(M.S), 1])
    policy = policy / policy_
    #policy = policy + (1.0 - policy.sum(axis = 1)).reshape([len(M.S), 1]) * temp
    print(policy.sum(axis = 1))
    M.set_policy(policy)
    
    M.output()
    exit()
    os.system('/home/zekunzhou/workspace/toyota_project/prism-4.4.beta-src/src/demos/run /home/zekunzhou/workspace/toyota_project/')
    os.system('/home/zekunzhou/workspace/toyota_project/prism-4.4.beta-src/bin/prism ./grid_world.pm ./grid_world.pctl')




