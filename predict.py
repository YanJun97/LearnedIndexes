from keras.models import Sequential, load_model
import numpy as np
import TwoStagesIndex

data = np.loadtxt('output2.txt', delimiter='\n')
std = data.std()
mean = data.mean()
model = load_model('mymodel2.h5')
#pos_pred = model.predict([(3103466683116293404 - mean) / std],batch_size=1)

max_err, min_err = 0, 0
data -= mean
data /= std
for i in range(len(data)):
    pos_pred = model.predict([(data[i] - mean) / std], batch_size=1)
    pos_pred = int(pos_pred)
    if pos_pred - i > max_err:
        max_err = pos_pred - i
    if i - pos_pred > min_err:
        min_err = i - pos_pred

print(max_err, min_err)

print()