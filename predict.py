from keras.models import Sequential, load_model
import numpy as np
from matplotlib import pyplot as plt


data = np.loadtxt('test.txt')
data = data[:, 0]
std = data.std()
mean = data.mean()
model = load_model('mymodel.h5')
#pos_pred = model.predict([(3103466683116293404 - mean) / std],batch_size=1)

max_err, min_err = 0, 0
data -= mean
data /= std

pos = model.predict(data)
for i in range(len(data)):
    # pos_pred = model.predict([(data[i] - mean) / std], batch_size=1)
    # pos_pred = int(pos_pred)
    pos_pred = pos[i]
    if pos_pred - i > max_err:
        max_err = pos_pred - i
    if i - pos_pred > min_err:
        min_err = i - pos_pred

print(max_err, min_err)
plt.plot(data, pos, 'ro')
plt.plot(data, [i for i in range(len(data))], 'bo')
plt.show()
print()