import numpy as np
import h5py
import json

#f = h5py.File('/Users/jiameng/Research@Bu/Toyota/DataSet.mat', 'r')
f = h5py.File('./DataSet.mat', 'r')
data = f.get('data')
data = np.mat(data)
action = f.get('Car1_act')
action = np.array(action)
no_car_dis = 500
no_car_v = -30
tra = []

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

def relative_value(car1, car2, car3):
    gap2 = car2 - car1
    gap3 = car3 - car1
    return gap2, gap3

def relative_position_(gap2, gap3):
    rel_position = np.zeros(8)
    
    #Same lane in the front
    if gap2[2]>=0 and abs(gap2[0])<= 2:
        rel_position[0] = 1
    #Same lane behind
    elif gap2[2]<=0 and abs(gap2[0])<= 2:
        rel_position[1] = 1
    #Left lane in the [-10, 10] zone
    if gap2[0]>2 and gap2[0]<4 and abs(gap2[2])<= 10:
        rel_position[2] = 1
    #Right lane in the [-10, 10] zone
    elif gap2[0]<-2 and gap2[0]>-4 and abs(gap2[2])<= 10:
        rel_position[3] = 1
        
    if gap3[2]>=0 and abs(gap2[0])<= 2:
        rel_position[4] = 1
    elif gap3[2]<=0 and abs(gap2[0])<= 2:
        rel_position[5] = 1
        
    if gap3[0]>2 and gap3[0]<4 and abs(gap3[2])<= 10:
        rel_position[6] = 1
    elif gap3[0]<-2 and gap3[0]>-4 and abs(gap3[2])<= 10:
        rel_position[7] = 1  
    return rel_position

def relative_position(gap2, gap3):
    rel_position = np.zeros(8)
    
    #In the front
    if gap2[2]>=0:
        rel_position[0] = 1
    #behind
    elif gap2[2]<=0:
        rel_position[1] = 1
    #Left lane
    if gap2[0]>2 and gap2[0]<4:
        rel_position[2] = 1
    #Right lane
    elif gap2[0]<-2 and gap2[0]>-4:
        rel_position[3] = 1
    
    if gap3[2]>=0:
        rel_position[4] = 1
    elif gap3[2]<=0:
        rel_position[5] = 1
    
    if gap3[0]>2 and gap3[0]<4:
        rel_position[6] = 1
    elif gap3[0]<-2 and gap3[0]>-4:
        rel_position[7] = 1
    return rel_position


def around(rel_position, gap2, gap3):
    dis = Neighbor()
    y_dis = np.zeros(2)
    y_v = np.zeros(2)
    x_dis = np.zeros(2)
    
    y_dis[0] = gap2[2]
    y_dis[1] = gap3[2]
    
    x_dis[0] = gap2[0]
    x_dis[1] = gap3[0]
    
    y_v[0] = gap2[3]
    y_v[1] = gap3[3]
    
    
# =============================================================================
#     if rel_position[0] == 1 & rel_position[4] == 1:
#         dis.f_dis = np.min(y_dis)
#         dis.b_dis = no_car_dis
#         dis.l = 0;
#         dis.r = 0;
#         dis.f_v = y_v[np.argmin(y_dis)]
#         dis.b_v = no_car_v
#         dis.l_v = no_car_v
#         dis.r_v = no_car_v
#         
#     elif rel_position[1] == 1 & rel_position[5] == 1:
#         dis.f_dis = no_car_dis
#         dis.b_dis = abs(np.max(y_dis))
#         dis.l = 0;
#         dis.r = 0;
#         dis.f_v = no_car_v
#         dis.b_v = y_v[np.argmax(y_dis)]
#         dis.l_v = no_car_v
#         dis.r_v = no_car_v
#         
#     elif rel_position[2] == 1 & rel_position[6] == 1:
#         dis.f_dis = no_car_dis
#         dis.b_dis = no_car_dis
#         dis.l = 1
#         dis.r = 0
#         dis.f_v = no_car_v
#         dis.b_v = no_car_v
#         dis.l_v = y_v[np.argmin(x_dis)]
#         dis.r_v = no_car_v
#         
#     elif rel_position[3] == 1 & rel_position[7] == 1:
#         dis.f_dis = no_car_dis
#         dis.b_dis = no_car_dis
#         dis.l = 0
#         dis.r = 1
#         dis.f_v = no_car_v
#         dis.b_v = no_car_v
#         dis.l_v = y_v[np.argmax(x_dis)]
#         dis.r_v = no_car_v
#     else:
# =============================================================================
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

for k in range(0, int(data.shape[1]/3)):
    gap2, gap3 = relative_value(data[:, 3*k], data[:, 3*k+1], data[:, 3*k+2])
    rel_position = relative_position(gap2, gap3)
    cur_state = around(rel_position, gap2, gap3)
    cur_state.lane_p = data[0, 3*k]
    tra.append(cur_state)
    

length = len(tra)
f_d = np.zeros(length)
b_d = np.zeros(length)
l = np.zeros(length)
r = np.zeros(length)
f_v = np.zeros(length)
b_v = np.zeros(length)
l_v = np.zeros(length)
r_v = np.zeros(length)
l_p = np.zeros(length)

for i in range(0, len(tra)):
    f_d[i] = tra[i].f_dis
    b_d[i] = tra[i].b_dis
    l[i] = tra[i].l
    r[i] = tra[i].r
    f_v[i] = tra[i].f_v
    b_v[i] = tra[i].b_v
    l_v[i] = tra[i].l_v
    r_v[i] = tra[i].r_v
    l_p[i] = tra[i].lane_p
    
r_V = r_v[r_v>0]

state_data ={"Front Distance": f_d.tolist(),
             "Rear Distance": b_d.tolist(),
             "Left": l.tolist(),
             "Right": r.tolist(),
             "Front Velocity": f_v.tolist(),
             "Rear Velocity": b_v.tolist(),
             "Left Velocity": l_v.tolist(),
             "Right Velocity": r_v.tolist(),
             "Lane Position": l_p.tolist(),
             "Action": action.tolist()}

with open('data.json', 'w') as fp: 
    json.dump(state_data, fp)

 
