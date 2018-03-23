import numpy as np
import scipy as sci


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
        '''

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
                if observation[i] >= self.threshes[i][j]:
                    continue
                else:
                    coord[i] = j
                    break
            if observation[i] >= self.threshes[i][-1]:
                coord[i] = len(self.threshes[i])
        return coord
