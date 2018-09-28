#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2018-08-31
@author: Keita Suzuki
'''

import numpy as np
import csv
import datetime

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer import serializers
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

READFILE_NAME = "qtable_category.csv"
FOLDER_PASS = "/Users/takashi/Documents/knowledge/qtable/"
SEED = 1145141919

batchsize = 128
max_epoch = 10

filename = []
category = []
data_set = []
np.random.seed(SEED)
if chainer.cuda.available:
    chainer.cuda.cupy.random.seed(SEED)

# csvファイルの読み込み
with open(READFILE_NAME, 'r') as o:
    dataReader = csv.reader(o)
    for row in dataReader:
        filename.append(row[0])
        category.append(row[1])
category = np.array(category, dtype=np.int)
for i in range(0,len(filename)):
    csvfile = str(FOLDER_PASS + filename[i])
    with open(csvfile, 'r') as o:
        dataReader = csv.reader(o)
        qtable = []
        for row in dataReader:
            qtable.append(row[3])
        qtable = np.array(qtable, dtype=np.float32)
        qtable = qtable.flatten()
        data_set.append([qtable,category[i]])
indices = np.arange(len(data_set))
i = np.random.choice(indices, 2500, replace=False)
np.random.shuffle(i)
data_set = np.array(data_set)
train_data = data_set[i[0:2000]]
train_X = []
train_Y = []
for j in range(0,len(train_data)):
    train_X.append(train_data[j][0])
    train_Y.append(train_data[j][1])
train_X = np.array(train_X, dtype=np.float32)
train_Y = np.array(train_Y, dtype=np.int32)

test_data = data_set[i[2000:]]
test_X = []
test_Y = []
for j in range(0, len(test_data)):
    test_X.append(test_data[j][0])
    test_Y.append(test_data[j][1])
test_X = np.array(test_X, dtype=np.float32)
test_Y = np.array(test_Y, dtype=np.int32)

# 2000個のtrainデータのうち1800個(8割)をTraining用データ、残りをValidation用データにする
train, valid = datasets.split_dataset_random(datasets.TupleDataset(train_X, train_Y), 1800, seed=0)
train_iter = iterators.SerialIterator(train, batchsize)
valid_iter = iterators.SerialIterator(valid, batchsize, repeat=False, shuffle=False)
# 500個のtestデータがtest用データになる
test = datasets.TupleDataset(test_X, test_Y)

class MLP(chainer.Chain):
    def __init__(self, n_mid_units=500, n_out=8):
        super(MLP, self).__init__()

        # パラメータを持つ層の登録
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(n_mid_units, n_mid_units)
            self.l3 = L.Linear(n_mid_units, n_out)

    def __call__(self, x):
        # データを受け取った際のforward計算を書く
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


model = L.Classifier(MLP())
optimizer = optimizers.SGD()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
filename = "result_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
trainer = training.Trainer(updater, (max_epoch, 'epoch'), out=filename)

trainer.extend(extensions.Evaluator(valid_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
trainer.extend(extensions.PlotReport(
    ['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(
    ['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
trainer.extend(extensions.dump_graph('main/accuracy'))
trainer.extend(extensions.ProgressBar())
trainer.run()

test_iter = iterators.MultiprocessIterator(test, batchsize, False, False)
test_evaluator = extensions.Evaluator(test_iter, model)
results = test_evaluator()
print('Test accuracy:', results['main/accuracy'])

accuracy_file = filename + "/Test_accuracy.txt"
with open(accuracy_file, mode="w") as w:
    w.write('%s' %results['main/accuracy'])
