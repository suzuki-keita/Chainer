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


READFILE_NAME = "category_2018-10-19.csv"
FOLDER_PASS = "/Users/takashi/Documents/knowledge/qtable_1/"
SEED = 1145141919

batchsize = 128
max_epoch = 500
datafiles = 6000

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
z = len(filename)
for i in range(0,len(filename)):
    csvfile = str(FOLDER_PASS + filename[i])
    with open(csvfile, 'r') as o:
        dataReader = csv.reader(o)
        qtable = np.zeros((5, 11, 11))
        x = 0
        y = 0
        for row in dataReader:
            a = int(row[2])
            qtable[a][x][y] = row[3]
            if a == 4:
                x = x+1
                if x == 11:
                    y = y+1
                    x = 0
        qtable = np.array(qtable, dtype=np.float32)
        data_set.append([qtable, category[i], csvfile])
    z = z - 1

indices = np.arange(len(data_set))

i = np.random.choice(indices, datafiles, replace=False)
np.random.shuffle(i)
data_set = np.array(data_set)
train_valid_datafiles = int(datafiles*0.8)
train_data = data_set[i[0:train_valid_datafiles]]
train_X = []
train_Y = []
train_csv = []
for j in range(0,len(train_data)):
    train_X.append(train_data[j][0])
    train_Y.append(train_data[j][1])
    train_csv.append([train_data[j][2], train_data[j][1]])
train_X = np.array(train_X, dtype=np.float32)
train_Y = np.array(train_Y, dtype=np.int32)

test_data = data_set[i[train_valid_datafiles:]]
test_X = []
test_Y = []
test_csv = []
for j in range(0, len(test_data)):
    test_X.append(test_data[j][0])
    test_Y.append(test_data[j][1])
    test_csv.append([test_data[j][2], test_data[j][1]])
test_X = np.array(test_X, dtype=np.float32)
test_Y = np.array(test_Y, dtype=np.int32)

# 2000個のtrainデータのうち1800個(8割)をTraining用データ、残りをValidation用データにする
train_datas = int(train_valid_datafiles*0.8)
train, valid = datasets.split_dataset_random(
    datasets.TupleDataset(train_X, train_Y), train_datas, seed=SEED)
train_iter = iterators.SerialIterator(train, batchsize)
valid_iter = iterators.SerialIterator(valid, batchsize, repeat=False, shuffle=False)
# 500個のtestデータがtest用データになる
test = datasets.TupleDataset(test_X, test_Y)

class MLP(chainer.Chain):
    def __init__(self, n_out=8):
        super(MLP, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 16, ksize=5, pad=2, nobias=True)
            self.conv1_2 = L.Convolution2D(None, 16, ksize=5, pad=2, nobias=True)
            self.conv2_1 = L.Convolution2D(None, 32, ksize=3, pad=1, nobias=True)
            self.conv2_2 = L.Convolution2D(None, 32, ksize=3, pad=1, nobias=True)
            self.fc1 = L.Linear(None, 512, nobias=True)
            self.fc2 = L.Linear(None, n_out, nobias=True)

    def __call__(self, x):
        conv1_1 = self.conv1_1(x)
        conv1_1 = F.relu(conv1_1)
        conv1_2 = self.conv1_2(conv1_1)
        conv1_2 = F.relu(conv1_2)
        pool1 = F.max_pooling_2d(conv1_2, ksize=2, stride=2)
        conv2_1 = self.conv2_1(pool1)
        conv2_1 = F.relu(conv2_1)
        conv2_2 = self.conv2_2(conv2_1)
        conv2_2 = F.relu(conv2_2)
        pool2 = F.max_pooling_2d(conv2_2, ksize=2, stride=2)
        fc1 = self.fc1(pool2)
        fc1 = F.relu(fc1)
        fc2 = self.fc2(fc1)
        return fc2


model = L.Classifier(MLP())
optimizer = optimizers.SGD()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
filename = "result_Q_CNN_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
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

#modelの保存(npz)
serializers.save_npz(filename + "/model.npz", model)

#optimizerの保存(npz)
serializers.save_npz(filename + "/model.state", optimizer)

train_file = filename + "/train.csv"
with open(train_file, mode="w") as w:
    writer = csv.writer(w, lineterminator='\n')
    writer.writerows(train_csv)

test_file = filename + "/test.csv"
with open(test_file, mode="w") as w:
    writer = csv.writer(w, lineterminator='\n')
    writer.writerows(test_csv)

accuracy_file = filename + "/Test_accuracy.txt"
with open(accuracy_file, mode="w") as w:
    w.write('%s' % results['main/accuracy'])

datas_file = filename + "/data.txt"
lists = [max_epoch, datafiles, train_datas,
         (train_valid_datafiles - train_datas), (datafiles - train_valid_datafiles), SEED]
with open(datas_file, mode="w") as w:
    w.write(
        'epoch:{0[0]}\nData:{0[1]}\nTraining_data:{0[2]}\nValidation_data:{0[3]}\ntest_data:{0[4]}\nSEED:{0[5]}'.format(lists))
