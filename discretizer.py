import numpy as np
import scipy as sci
import ast

class discretizer:
    def __init__(self):
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
                     [0, 10, 20, 30, 40],  # car1 velocity
                
                     [-41],  # car2 lane
                     [-30, -10, -3, 3, 10, 30],  # car2 distance
                     [-10, 0, 10],  # car2 speed
                         
                     [-41],  # car3 lane
                     [-30, -10, -3, 3, 10, 30],  # car3 distance
                     [-10, 0, 10],  # car3 speed
                     ]  # list of threshes for each dimension
        self.grids = []
        for i in self.threshes:
            # number of intervals in each dimension
            self.grids.append(len(i) + 1)
        # [front dist, rear dist, left, right, front speed, rear speed, left speed, right speed, lane pos]

    def build_discretizer(self, grids, threshes):
        assert len(grids) == len(threshes)
        for i in range(len(grids)):
            if(grids[i] != len(threshes[i])):
                raise ValueError("threshold number conflict grids")
        self.grids = grids
        self.threshes = threshes

    def observation_to_coord(self, observation):
        # translate observation to coordinates
        coord = np.zeros([len(self.grids)])
        for i in range(len(observation)):
            for j in range(len(self.threshes[i])):
                if observation[i] > self.threshes[i][j]:
                    continue
                else:
                    coord[i] = j
                    break
            if observation[i] >= self.threshes[i][-1]:
                coord[i] = len(self.threshes[i])
        return coord.tolist()

    def build_transitions(self, path = './data/data', freq = 20):
        # Preprocess the data set file which contains trajectories
        # Each trajectory is a list in which each element in a list is a list of time step, dict of observations and ...
        # This method translates the observations to coords
        tuples = []
        last_tuple = list()
        index = 0
        
        file_i = open(path, 'r')
        print("read list file")
        for line_str in file_i.readlines():
            line = ast.literal_eval(line_str)
            time = line[0]
            
            #From coord
            observation = line[1]
            coord = self.observation_to_coord(observation)
            
            #To coord
            observation_ = line[-1]
            coord_ = self.observation_to_coord(observation_)
            
            #Judge the action between coord, action = [change lane\in {0, 1}, change speed \in{0, 1, 2}]
            action = [0, 0]
            
            #If lane is changed
            if coord[0] != coord_[0]:
                action[0] = 1
            else:
                action[0] = 0
            
            #If speed increases
            if coord_[1] - coord[1] >= 1:
                action[1] = 2
            elif coord_[1] - coord[1] <= -1:
                action[1] = 0
            else:
                action[1] = 1
        
            #If there is no changes in the environment, then discard this record
            tuple = [i for i in coord + action + coord_]
            if last_tuple == tuple and action == [0, 1] and time%freq != 0:
                continue
            last_tuple = [i for i in tuple]
            
            tuples.append(
                          ('[' +
                           str(index) +
                           ', ' +
                           str(coord) +
                           ', ' +
                           str(action) +
                           ', ' +
                           str(coord_) +
                           ']\n'))
            index += 1

        file_i.close()
        
        file_o = open('./data/demo', 'w')
        for line_str in tuples:
            file_o.write(line_str)
        file_o.close()

if __name__ == "__main__":
    obs = discretizer()
    obs.build_transitions()






