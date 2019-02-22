import TwoStagesIndex
import numpy as np
import time


data = np.loadtxt('output2.txt', delimiter='\n')
mean, std = data.mean(), data.std()
data -= mean
data /= std

time1 = time.time()
model = TwoStagesIndex.RecursiveModelIndexes(data, [i for i in range(len(data))], [1, 2000], 100)
time2 = time.time()
model.train()
time3 = time.time()

print(time1, time2, time3)
key = 3103216050784029464
pos = model.predict([(key-mean)/std])
print(pos)
