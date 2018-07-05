import json

def preprocess_dict(path = './data/data.json'):
    file = open(path, 'r')
    dic = json.load(file)
    
    '''
    print(len(dic["Car1_Position"]))
    print(len(dic["Car1_Lane_Position"]))
    print(len(dic["Car1_Velocity"]))
    print(len(dic["Car2_Lane_Position"]))
    print(len(dic["Car2_Distance"]))
    print(len(dic["Car2_Velocity"]))
    print(len(dic["Car3_Lane_Position"]))
    print(len(dic["Car3_Distance"]))
    print(len(dic["Car3_Velocity"]))
    print(len(dic['Action']))
    '''

    time = 0
    trajectory = -1
    demo = {}
    for index in range(len(dic["Car1_Position"]) - 1):
        if dic["Car1_Position"][index] < 5.0 or dic["Car1_Lane_Position"][index] < 5.0:
            time = 0
            continue
        if time == 0:
            trajectory += 1
            demo[trajectory] = []
        demo[trajectory].append({})
        demo[trajectory][-1]['time'] = time

        demo[trajectory][-1]['from'] = {}
        demo[trajectory][-1]['from']["Car1_Position"] = dic["Car1_Position"][index]
        demo[trajectory][-1]['from']["Car1_Lane_Position"] = dic['Rear Distance'][index]
        demo[trajectory][-1]['from']["Car1_Velocity"] = dic["Car1_Velocity"][index]
        demo[trajectory][-1]['from']["Car2_Lane_Position"] = dic["Car2_Lane_Position"][index]
        demo[trajectory][-1]['from']["Car2_Distance"] = dic["Car2_Distance"][index]
        demo[trajectory][-1]['from']["Car2_Velocity"] = dic["Car2_Velocity"][index]
        demo[trajectory][-1]['from']["Car3_Lane_Position"] = dic["Car3_Lane_Position"][index]
        demo[trajectory][-1]['from']["Car3_Distance"] = dic["Car3_Distance"][index]
        demo[trajectory][-1]['from']["Car3_Velocity"] = dic["Car3_Velocity"][index]

        demo[trajectory][-1]['Action'] = dic['Car1_Action'][index]

        demo[trajectory][-1]['to'] = {}
        demo[trajectory][-1]['to']["Car1_Position"] = dic["Car1_Position"][index + 1]
        demo[trajectory][-1]['to']["Car1_Lane_Position"] = dic['Rear Distance'][index + 1]
        demo[trajectory][-1]['to']["Car1_Velocity"] = dic["Car1_Velocity"][index + 1]
        demo[trajectory][-1]['to']["Car2_Lane_Position"] = dic["Car2_Lane_Position"][index + 1]
        demo[trajectory][-1]['to']["Car2_Distance"] = dic["Car2_Distance"][index + 1]
        demo[trajectory][-1]['to']["Car2_Velocity"] = dic["Car2_Velocity"][index + 1]
        demo[trajectory][-1]['to']["Car3_Lane_Position"] = dic["Car3_Lane_Position"][index + 1]
        demo[trajectory][-1]['to']["Car3_Distance"] = dic["Car3_Distance"][index + 1]
        demo[trajectory][-1]['to']["Car3_Velocity"] = dic["Car3_Velocity"][index + 1]

        time += 1
     
    file.close()
    with open('./data/demo.json', 'w') as f:
        json.dump(demo, f)

def preprocess_list(path = './data/data.json'):
    file = open(path, 'r')
    dic = json.load(file)

    '''    
    print(len(dic["Car1_Position"]))
    print(len(dic["Car1_Lane_Position"]))
    print(len(dic["Car1_Velocity"]))
    print(len(dic["Car2_Lane_Position"]))
    print(len(dic["Car2_Distance"]))
    print(len(dic["Car2_Velocity"]))
    print(len(dic["Car3_Lane_Position"]))
    print(len(dic["Car3_Distance"]))
    print(len(dic["Car3_Velocity"]))
    print(len(dic['Action']))
    '''

    time = 0
    trajectory = -1
    demo = []
    print(len(dic["Car1_Position"]))
    for index in range(len(dic["Car1_Position"]) - 1):
        if abs(dic["Car2_Distance"][index]) < 1.0 or abs(dic["Car3_Distance"][index] < 1.0) or abs(dic["Car1_Position"][index] - dic["Car1_Position"][index + 1]) > 5:
            time = 0
            continue
        if time == 0:
            trajectory += 1
            demo.append([])
        demo[trajectory].append([])

        demo[trajectory][-1].append(time)

        demo[trajectory][-1].append([])
        demo[trajectory][-1][-1].append(dic["Car1_Lane_Position"][index])
        #demo[trajectory][-1][-1].append(dic["Car1_Position"][index])
        demo[trajectory][-1][-1].append(dic["Car1_Velocity"][index])
        demo[trajectory][-1][-1].append(dic["Car2_Lane_Position"][index])
        demo[trajectory][-1][-1].append(dic["Car2_Distance"][index])
        demo[trajectory][-1][-1].append(dic["Car2_Velocity"][index])
        demo[trajectory][-1][-1].append(dic["Car3_Lane_Position"][index])
        demo[trajectory][-1][-1].append(dic["Car3_Distance"][index])
        demo[trajectory][-1][-1].append(dic["Car3_Velocity"][index])
        
        demo[trajectory][-1].append(dic['Car1_Action'][index])

        demo[trajectory][-1].append([])
        demo[trajectory][-1][-1].append(dic["Car1_Lane_Position"][index + 1])
        #demo[trajectory][-1][-1].append(dic["Car1_Lane_Position"][index + 1])
        demo[trajectory][-1][-1].append(dic["Car1_Velocity"][index + 1])
        demo[trajectory][-1][-1].append(dic["Car2_Lane_Position"][index + 1])
        demo[trajectory][-1][-1].append(dic["Car2_Distance"][index + 1])
        demo[trajectory][-1][-1].append(dic["Car2_Velocity"][index + 1])
        demo[trajectory][-1][-1].append(dic["Car3_Lane_Position"][index + 1])
        demo[trajectory][-1][-1].append(dic["Car3_Distance"][index + 1])
        demo[trajectory][-1][-1].append(dic["Car3_Velocity"][index + 1])

        time += 1
    print("Total number of trajectories")
    print(len(demo))
    print("Example:")
    print(demo[-1][-1])
    file.close()
    file = open('./data', 'w')
    for i in range(len(demo)):
        for j in range(len(demo[i])):
            file.write(str(demo[i][j]) + '\n')

