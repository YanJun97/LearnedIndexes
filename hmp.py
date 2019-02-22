import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,normalization,Lambda
from keras import optimizers


s_gd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=False)


data = np.loadtxt('output2.txt', delimiter='\n').reshape((-1, 1))
labels = np.zeros(data.shape, dtype='int32')
mean = np.mean(data)
std = np.std(data)
for i in range(len(data)):
    labels[i] = i

data -= mean
data /= std

model = Sequential()
model.add(Dense(1, dtype='float32',activation='relu', input_dim=1, use_bias=True))
#model.add(normalization.BatchNormalization())
model.add(Dense(1, use_bias=True))

model.compile(optimizer=s_gd, loss='mae')
model.fit(data, labels, epochs=5, batch_size=256, verbose=1, shuffle=True)
model.save('mymodel2.h5')


