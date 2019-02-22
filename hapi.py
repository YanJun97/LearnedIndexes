import numpy as np


data = np.loadtxt('output1.txt', dtype='int64', delimiter='\n')
data.sort()
np.savetxt('output2.txt', data, fmt='%ld', delimiter='\n')

