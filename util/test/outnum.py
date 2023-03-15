import random
import json
import datetime
train=[random.randint(1,100) for i in range(20)]
train=sorted(train[:])

test=[random.randint(1,100) for i in range(20)]
test=sorted(test[:])

def out(list,file):
    jsonImg = json.dumps(list)
    f1 = open(file, 'w')
    f1.write(jsonImg)
    f1.close()

ans = [train,test]
date=str(datetime.datetime.now().strftime('%Y-%m-%d-%X')).replace(":","-")
filename = date+'_train.txt'
out(ans,'./'+filename)
# print(train)