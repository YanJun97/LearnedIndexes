from sklearn import linear_model
import numpy as np


x = np.loadtxt('output2.txt', dtype='int64', delimiter='\n').reshape((-1, 1))
index = [i for i in range(len(x))]
print(len(x))

model = linear_model.Ridge(alpha=0.5)
model.fit(x, index)

max_err, min_err = 0, 0

# for i in range(len(x)):
#     temp = model.predict([x[i]])
#     if temp - i > max_err:
#         max_err = int(temp - i)
#     if i - temp > min_err:
#         min_err = int(i - temp)
# print(max_err, min_err)

# print W, b
print(model.coef_, model.intercept_)
