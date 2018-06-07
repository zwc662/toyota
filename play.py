import numpy as np
import scipy.sparse as sparse
import os
import ast

"""
Make sure that 'policy' is in the same path as this 'play.py' file.

1) Instantiate the play class in this file;

2) Call play.run(observation) method and it shall return an action.

The variable 'observation' should be a list.

Each element of the list is a list of observed data (floating numbers) of one car, which is in the same sequence as the data in the raw data file.

Since there are 3 cars, there should be three lists in the 'observation' list.

The returned action is an integer chosen from {0, 1, 2, 3, 4}, in which 0 means doing nothing while others have the same meanings as the actions in the raw data file.

 For example:
 #!/usr/bin/env python3 (or python)
 from play import play
 play = play()
 a = play.run([[1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1]])

"""


class play:
    def __init__(self):
        # Transfer car data to array
        observation = list()
        policy = list()
        state = 0
        action = 0
        
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
        for i in self.threshes:
            # number of intervals in each dimension
            self.grids.append(len(i) + 1)

    def policy(self, path = 'policy'):
        file = open(path, 'r')
        lines = file.readlines()
        for line in lines:
            actions = [float(a) for a in line.split(' ')[:-1]]
            actions = np.array(actions)
            self.policy.append(np.argmax(actions))
        file.clost()
        return self.policy

            
    def act(self, data, policy = None):
        if policy is not None:
            self.policy = policy

        observation = self.rawdata(data)

        state = self.observation_to_index(observation)
        
        return self.policy(state)

    
    def rawdata(self, data):
        #data = [car1 lane pos, car1 y pos, car1 z pos, car1 v, \
        #        car2 lane pos, car2 y pos, car2 z pos, car2 v, \
        #        car3 lane pos, car3 y pos, car3 z pos, car3 v]

        l_p_1 = data[0]
        v_1 = data[3]
        l_p_2 = data[4]
        d_2 = data[5] - data[1]
        v_2 = data[7] 
        l_p_3 = data[8]
        d_3 = data[9] - data[1]
        v_3 = data[11]

        return [l_p_1, v_1, l_p_2, d_2, v_2, l_p_3, d_3, v_3]


    def observation_to_coord(self, observation):
        # translate observation to coordinates
        coord = np.zeros([len(self.grids)]).astype(int)
        for i in range(len(observation)):
            for j in range(len(self.threshes[i])):
                if observation[i] >= self.threshes[i][j]:
                    continue
                else:
                    coord[i] = int(j)
                    break
            if observation[i] >= self.threshes[i][-1]:
                coord[i] = len(self.threshes[i])
        return coord

    def observation_to_index(self, observation):
        coord = self.observation_to_coord(observation)
        index = self.coord_to_index(coord)
        return int(index)


    def index_to_coord(self, index):
        coord = self.grids[:]
        for i in range(len(self.grids)):
            coord[len(self.grids) - 1 - i] = index%self.grids[len(self.grids) - 1 - i]
            index = int(index/self.grids[len(self.grids) - 1 - i])
        return coord

    def coord_to_index(self, coord):
        # Translate observation to coordinate by calling the grids
        # Then translate the coordinate to index
        index = 0
        base = 1
        for i in range(len(coord)):
            index += coord[len(coord) - 1 - i] * base 
            base *= self.grids[len(self.grids) - 1 - i]
        return int(index)

