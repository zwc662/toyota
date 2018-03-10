import json

def preprocess_dict(path):
    file = open(path, 'r')
    dic = json.load(file)
    
    print(len(dic['Front Distance']))
    print(len(dic['Rear Distance']))
    print(len(dic['Left']))
    print(len(dic['Right']))
    print(len(dic['Front Velocity']))
    print(len(dic['Rear Velocity']))
    print(len(dic['Left Velocity']))
    print(len(dic['Right Velocity']))
    print(len(dic['Lane Position']))
    print(len(dic['Action']))
    
    time = 0
    trajectory = -1
    demo = {}
    for index in range(len(dic['Front Distance']) - 1):
        if dic['Front Distance'][index] < 5.0 or dic['Rear Distance'][index] < 5.0:
            time = 0
            continue
        if time == 0:
            trajectory += 1
            demo[trajectory] = []
        demo[trajectory].append({})
        demo[trajectory][-1]['time'] = time

        demo[trajectory][-1]['from'] = {}
        demo[trajectory][-1]['from']['Front Distance'] = dic['Front Distance'][index]
        demo[trajectory][-1]['from']['Rear Distance'] = dic['Rear Distance'][index]
        demo[trajectory][-1]['from']['Left'] = dic['Left'][index]
        demo[trajectory][-1]['from']['Right'] = dic['Right'][index]
        demo[trajectory][-1]['from']['Front Velocity'] = dic['Front Velocity'][index]
        demo[trajectory][-1]['from']['Rear Velocity'] = dic['Rear Velocity'][index]
        demo[trajectory][-1]['from']['Left Velocity'] = dic['Left Velocity'][index]
        demo[trajectory][-1]['from']['Right Velocity'] = dic['Right Velocity'][index]
        demo[trajectory][-1]['from']['Lane Position'] = dic['Lane Position'][index]

        demo[trajectory][-1]['Action'] = dic['Action'][index]

        demo[trajectory][-1]['to'] = {}
        demo[trajectory][-1]['to']['Front Distance'] = dic['Front Distance'][index + 1]
        demo[trajectory][-1]['to']['Rear Distance'] = dic['Rear Distance'][index + 1]
        demo[trajectory][-1]['to']['Left'] = dic['Left'][index + 1]
        demo[trajectory][-1]['to']['Right'] = dic['Right'][index + 1]
        demo[trajectory][-1]['to']['Front Velocity'] = dic['Front Velocity'][index + 1]
        demo[trajectory][-1]['to']['Rear Velocity'] = dic['Rear Velocity'][index + 1]
        demo[trajectory][-1]['to']['Left Velocity'] = dic['Left Velocity'][index + 1]
        demo[trajectory][-1]['to']['Right Velocity'] = dic['Right Velocity'][index + 1]
        demo[trajectory][-1]['to']['Lane Position'] = dic['Lane Position'][index + 1]

        time += 1
     
    file.close()
    with open('./data/demo.json', 'w') as f:
        json.dump(demo, f)

def preprocess_list(path):
    file = open(path, 'r')
    dic = json.load(file)
    
    print(len(dic['Front Distance']))
    print(len(dic['Rear Distance']))
    print(len(dic['Left']))
    print(len(dic['Right']))
    print(len(dic['Front Velocity']))
    print(len(dic['Rear Velocity']))
    print(len(dic['Left Velocity']))
    print(len(dic['Right Velocity']))
    print(len(dic['Lane Position']))
    print(len(dic['Action']))
    
    time = 0
    trajectory = -1
    demo = []
    for index in range(len(dic['Front Distance']) - 1):
        if dic['Front Distance'][index] < 5.0 or dic['Rear Distance'][index] < 5.0 or abs(dic['Front Distance'][index] - dic['Front Distance'][index + 1]) > 5 or abs(dic['Rear Distance'][index] - dic['Rear Distance'][index + 1]) > 5:
            time = 0
            continue
        if time == 0:
            trajectory += 1
            demo.append([])
        demo[trajectory].append([])
        demo[trajectory][-1].append(time)
        demo[trajectory][-1].append([])
        demo[trajectory][-1][-1].append(dic['Front Distance'][index])
        demo[trajectory][-1][-1].append(dic['Rear Distance'][index])
        demo[trajectory][-1][-1].append(dic['Left'][index])
        demo[trajectory][-1][-1].append(dic['Right'][index])
        demo[trajectory][-1][-1].append(dic['Front Velocity'][index])
        demo[trajectory][-1][-1].append(dic['Rear Velocity'][index])
        demo[trajectory][-1][-1].append(dic['Left Velocity'][index])
        demo[trajectory][-1][-1].append(dic['Right Velocity'][index])
        demo[trajectory][-1][-1].append(dic['Lane Position'][index])
        
        demo[trajectory][-1].append(dic['Action'][index])
        demo[trajectory][-1].append([])
        
        demo[trajectory][-1][-1].append(dic['Front Distance'][index + 1])
        demo[trajectory][-1][-1].append(dic['Rear Distance'][index + 1])
        demo[trajectory][-1][-1].append(dic['Left'][index + 1])
        demo[trajectory][-1][-1].append(dic['Right'][index + 1])
        demo[trajectory][-1][-1].append(dic['Front Velocity'][index + 1])
        demo[trajectory][-1][-1].append(dic['Rear Velocity'][index + 1])
        demo[trajectory][-1][-1].append(dic['Left Velocity'][index + 1])
        demo[trajectory][-1][-1].append(dic['Right Velocity'][index + 1])
        demo[trajectory][-1][-1].append(dic['Lane Position'][index + 1])

        time += 1
            
    file.close()
    file = open('./data/demo', 'w')
    for i in range(len(demo)):
        for j in range(len(demo[i])):
            file.write(str(demo[i][j]) + '\n')

