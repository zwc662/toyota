import json

def preprocess_list(path):
    file = open(path, 'r')
    dic = json.load(file)

    time = 0
    trajectory = -1
    demo = []
    print(len(dic['Car2_Distance']) - 1)
    for index in range(len(dic['Car2_Distance']) - 1):
        if (abs(dic['Car1_Lane_Position'][index] - dic['Car2_Lane_Position'][index]) <= 1.0 and dic['Car2_Distance'][index] < 3.0) or \
           (abs(dic['Car1_Lane_Position'][index] - dic['Car3_Lane_Position'][index]) <= 1.0 and dic['Car3_Distance'][index] < 3.0) or \
           abs(dic['Car1_Position'][index] - dic['Car1_Position'][index + 1]) > 10:
            time = 0
            continue
        if time == 0:
            trajectory += 1
            demo.append([])
        demo[trajectory].append([])
        demo[trajectory][-1].append(time)
        demo[trajectory][-1].append([])
        demo[trajectory][-1][-1].append(dic['Car1_Lane_Position'][index])
        demo[trajectory][-1][-1].append(dic['Car1_Velocity'][index])
        demo[trajectory][-1][-1].append(dic['Car2_Lane_Position'][index])
        demo[trajectory][-1][-1].append(dic['Car2_Distance'][index])
        demo[trajectory][-1][-1].append(dic['Car2_Velocity'][index])
        demo[trajectory][-1][-1].append(dic['Car3_Lane_Position'][index])
        demo[trajectory][-1][-1].append(dic['Car3_Distance'][index])
        demo[trajectory][-1][-1].append(dic['Car3_Velocity'][index])
        
        #demo[trajectory][-1].append(dic['Action'][index])
        demo[trajectory][-1].append([])
        
        demo[trajectory][-1][-1].append(dic['Car1_Lane_Position'][index+1])
        demo[trajectory][-1][-1].append(dic['Car1_Velocity'][index+1])
        demo[trajectory][-1][-1].append(dic['Car2_Lane_Position'][index+1])
        demo[trajectory][-1][-1].append(dic['Car2_Distance'][index+1])
        demo[trajectory][-1][-1].append(dic['Car2_Velocity'][index+1])
        demo[trajectory][-1][-1].append(dic['Car3_Lane_Position'][index+1])
        demo[trajectory][-1][-1].append(dic['Car3_Distance'][index+1])
        demo[trajectory][-1][-1].append(dic['Car3_Velocity'][index+1])

        time += 1
    print(len(demo))
    file.close()
    file = open('./data/data_v1', 'w')
    for i in range(len(demo)):
        for j in range(len(demo[i])):
            file.write(str(demo[i][j]) + '\n')
    file.close()

if __name__ == "__main__":
    preprocess_list("./data/data_v1.json")
