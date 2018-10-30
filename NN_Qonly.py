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
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

READFILE_NAME = "category_2018-10-19.csv"
FOLDER_PASS = "/Users/takashi/Documents/knowledge/qtable_1/"
SEED = 1145141919

batchsize = 128
max_epoch = 1000
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
for i in range(0, len(filename)):
    csvfile = str(FOLDER_PASS + filename[i])
    with open(csvfile, 'r') as o:
        dataReader = csv.reader(o)
        qtable = []
        for row in dataReader:
            qtable.append(row[3])
        qtable = np.array(qtable, dtype=np.float32)
        qtable = qtable.flatten()
        data_set.append([qtable, category[i], csvfile])
indices = np.arange(len(data_set))
i = np.random.choice(indices, datafiles, replace=False)
np.random.shuffle(i)
#データの内、8割をtrain-validationデータ,2割をtestデータとする
train_valid_datafiles = int(datafiles*0.8)
data_set = np.array(data_set)
train_data = data_set[i[0:train_valid_datafiles]]
train_X = []
train_Y = []
train_csv = []
for j in range(0, len(train_data)):
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

# trainデータのうち8割をTraining用データ、残りをValidation用データにする
train_datas = int(train_valid_datafiles*0.8)
train, valid = datasets.split_dataset_random(
    datasets.TupleDataset(train_X, train_Y), train_datas, seed=SEED)
train_iter = iterators.SerialIterator(train, batchsize)
valid_iter = iterators.SerialIterator(
    valid, batchsize, repeat=False, shuffle=False)
# 2割のtestデータがtest用データになる
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
filename = "result_Qonly_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
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
