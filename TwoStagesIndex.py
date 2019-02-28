import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers


def init_nn(stage):
    if stage == 1:
        model = Sequential()
        model.add(Dense(units=2, input_dim=1, activation='relu', use_bias=True, dtype='float32'))
        model.add(Dense(1, use_bias=True, dtype='float32'))
        s_gd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=False)
        model.compile(optimizer=s_gd, loss='mae')
        return model
    elif stage == 2:
        model = Sequential()
        model.add(Dense(1, input_shape=(1,), use_bias=True))
        model.compile(optimizer='sgd', loss='mse')
        return model
    else:
        raise Exception('wrong stages!')


class RecursiveModelIndexes:

    def __init__(self, X, y, stages, threshold):
        self.x = X
        self.y = y
        self.datasize = len(X)
        self.stages = stages
        # self.compexity = NN_complexity
        self.threshold = threshold
        self.models = []
        self.models.append([init_nn(1)])
        self.models.append([])
        self.max_errs = [0 for _ in range(stages[-1])]
        for i in range(self.stages[1]):
            self.models[1].append(init_nn(2))

    def train(self):
        X = self.x
        y = self.y
        num = self.stages[1]
        tmp_records = [[] for _ in range(num)]
        tmp_labels = [[] for _ in range(num)]
        # stage1
        self.models[0][-1].fit(X, y, batch_size=256, epochs=5)
        for i in range(len(X)):
            # TODO:i 考虑输出是否可能为负数，或超过data size，应如何处理√
            p = self.models[0][-1].predict(X[i].reshape((-1,1)))
            p = max(0, p)
            p = int(p * num / self.datasize)
            p = min(p, num - 1)
            tmp_records[p].append(X[i].reshape((-1,1)))
            tmp_labels[p].append(y[i])
        # stage2
        for i in range(self.stages[1]):
            self.models[1][i].fit(tmp_records[i][0], tmp_records[i][1], batch_size=32, epochs=1)
            # 可以加个metric
            self.max_errs[i] = self.models[1][i].evaluate(tmp_records[i][0], tmp_records[i][1],)
            # TODO:ii 若max_err超过threshold,则将该模型换位b树
        return 0

    def predict(self, key):
        # TODO:iii 同i
        p = self.models[0][-1].predict(key)
        p = max(0, p)
        p = int(p * self.stages[1] / self.datasize)
        p = min(p, self.stages[-1] - 1)
        pos = self.models[1][p].predict(key)

        return pos, p

    def index(self, key):
        pos, p = self.predict(key)
        # TODO:iv 在数组上查找x,用二分查找，第一个分裂点取pos,（应该将数组引用传进来）
        return pos



