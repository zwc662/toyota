#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 09:23:37 2018

@author: jiameng
"""

import numpy as np

class Neighbor:
    def _init_(self, f_dis, b_dis, l, r, f_v, b_v, l_v, r_v, lane_p):
        self.f_dis = f_dis
        self.b_dis = b_dis
        self.l = l
        self.r = r
        self.f_v = f_v
        self.b_v = b_v
        self.l_v = l_v
        self.r_v = r_v
        self.lane_p = lane_p

class Observation:
    def __init__(self, car1, car2, car3):
        # Transfer car data to array
        self.car_central = np.array(car1)
        self.car2 = np.array(car2)
        self.car3 = np.array(car3)
        self.lane_p = car1[0]
        
        #set no_car_distance and no_car_velocity
        self.no_car_dis = 500
        self.no_car_v = -30
        
    def relative_value(self):
        #Compute the relative distance and velocity between cars
        gap2 = self.car2 - self.car_central
        gap3 = self.car3 - self.car_central
        return gap2, gap3
    
    def relative_position(self):
        #Get the relative values for cars
        gap2, gap3 = self.relative_value()
        
        #initialize position around the central car
        rel_position = np.zeros(8)
        
        #if car2 is in front or back of the central car
        if gap2[2]>=0 and abs(gap2[0])<= 2:
            rel_position[0] = 1
        elif gap2[2]<=0 and abs(gap2[0])<= 2:
            rel_position[1] = 1
            
        # if car2 is in left or right of the central car
        if gap2[0]>2 and gap2[0]<4 and abs(gap2[2])<= 10:
            rel_position[2] = 1
        elif gap2[0]<-2 and gap2[0]>-4 and abs(gap2[2])<= 10:
            rel_position[3] = 1
        
        # if car3 is in front or back of the central car
        if gap3[2]>=0 and abs(gap2[0])<= 2:
            rel_position[4] = 1
        elif gap3[2]<=0 and abs(gap2[0])<= 2:
            rel_position[5] = 1
         
        # if car3 is in left or right of the central car
        if gap3[0]>2 and gap3[0]<4 and abs(gap3[2])<= 10:
            rel_position[6] = 1
        elif gap3[0]<-2 and gap3[0]>-4 and abs(gap3[2])<= 10:
            rel_position[7] = 1  
        return rel_position
    
    def around(self):
        #Get the relative position
        gap2, gap3 = self.relative_value()
        rel_position = self.relative_position()
        
        #Set no_car_distance and no_car_velocity
        no_car_dis = self.no_car_dis
        no_car_v = self.no_car_v
        
        #Initilize Neighbor()
        dis = Neighbor()
        
        #Extract from raw data
        y_dis = np.zeros(2)
        y_v = np.zeros(2)
        x_dis = np.zeros(2)
        
        y_dis[0] = gap2[2]
        y_dis[1] = gap3[2]
        
        x_dis[0] = gap2[0]
        x_dis[1] = gap3[0]
        
        y_v[0] = gap2[3]
        y_v[1] = gap3[3]
        
        
        front_dis = np.zeros(2)
        rear_dis = np.zeros(2)
        left_dis = np.zeros(2)
        right_dis = np.zeros(2)
            
        front_dis[0] = rel_position[0]*y_dis[0] 
        front_dis[1] = rel_position[4]*y_dis[1]
        rear_dis[0] = rel_position[1]*y_dis[0]
        rear_dis[1] = rel_position[5]*y_dis[1]
        left_dis[0] = rel_position[2]*x_dis[0]
        left_dis[1] = rel_position[6]*x_dis[1]
        right_dis[0] = rel_position[3]*x_dis[0]
        right_dis[1] = rel_position[7]*x_dis[1]
            
        if np.max(front_dis) == 0:
            dis.f_dis = no_car_dis
            dis.f_v = -no_car_v
        else:
            dis.f_dis = np.min(front_dis[front_dis>0])
            dis.f_v = y_v[np.argmin(front_dis[front_dis>0])]
                
        if np.min(rear_dis) == 0:
            dis.b_dis = no_car_dis
            dis.b_v = no_car_v
        else:
            dis.b_dis = abs(np.min(rear_dis))
            dis.b_v = y_v[np.argmin(rear_dis)]
        
        if rel_position[2] == 1 or rel_position[6] == 1:
            dis.l = 1
            dis.l_v = y_v[np.argmin(left_dis[left_dis>0])]
        else:
            dis.l = 0
            dis.l_v = no_car_v
            
        if rel_position[3] == 1 or rel_position[7] == 1:
            dis.r = 1
            dis.r_v = abs(y_v[np.argmin(right_dis)])
        else:
            dis.r = 0
            dis.r_v = no_car_v
            
        return dis
    
    def convert_rawdata(self):
        #Get current around state
        cur_state = self.around()
        
        #Get current lane_position of the central car
        cur_state.lane_p = self.lane_p
        return [cur_state.f_dis, cur_state.b_dis, cur_state.l, cur_state.r, 
                cur_state.f_v, cur_state.b_v, cur_state.l_v, cur_state.r_v, 
                cur_state.lane_p]
        
