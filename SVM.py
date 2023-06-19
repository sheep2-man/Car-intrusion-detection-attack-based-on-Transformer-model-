import torch.nn as nn
import numpy as np
import config
import torch.nn as nn
import torch.optim as optim
import config
import torch
import time
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score,recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from getdata import MyData
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import svm
train_set = MyData(training=0)
test_set=MyData(training=2)

train_features = []
train_label = []
test_features = []
test_label = []

for att, label in train_set:
    train_features.append(att)
    train_label.append(label)

for att, label in test_set:
    test_features.append(att)
    test_label.append(label)

train_features = np.array(train_features)
train_label = np.array(train_label)
test_features = np.array(test_features)
test_label = np.array(test_label)

clf = svm.LinearSVC(C=1,  max_iter=100)
clf.fit(train_features, train_label)

# Test on Training data
train_result = clf.predict(train_features)
precision = sum(train_result == train_label) / train_label.shape[0]
print('Training precision: ', precision)

# Test on test data
test_result = clf.predict(test_features)
precision = sum(test_result == test_label) / test_label.shape[0]
print('Test precision: ', precision)
print('accuracy:{}'.format(accuracy_score(test_label, test_result)))
print('precision:{}'.format(precision_score(test_label, test_result, average='micro')))
print('recall:{}'.format(recall_score(test_label, test_result, average='micro')))
print('f1-score:{}'.format(f1_score(test_label, test_result, average='micro')))

