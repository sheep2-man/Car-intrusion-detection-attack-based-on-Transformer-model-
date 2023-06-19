from torchvision import datasets, transforms
#引入numpy计算矩阵
import numpy as np
#引入模型评估指标 accuracy_score
from sklearn.metrics import accuracy_score
import torch
#引入进度条设置以及时间设置
from tqdm import tqdm
from getdata import MyData
import time
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score,recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier
# 定义KNN函数
def KNN(train_x, train_y, test_x, test_y, k):
    #获取当前时间
    since = time.time()
    #可以将m,n理解为求其数据个数，属于torch.tensor类
    # m = test_x.size(0)
    # n = train_x.size(0)
    train_x, train_y, test_x, test_y = torch.tensor(train_x).cuda(), torch.tensor(train_y).cuda(), torch.tensor(
        test_x).cuda(), torch.tensor(
        test_y).cuda()
    # train_x,train_y,test_x,test_y=torch.tensor(train_x),torch.tensor(train_y),torch.tensor(test_x),torch.tensor(test_y)
    m=len(test_x)
    n=len(train_x)
    # 计算欧几里得距离矩阵，矩阵维度为m*n；
    # print("计算距离矩阵")

    #test,train本身维度是m*1, **2为对每个元素平方，sum(dim=1，对行求和；keepdim =True时保持二维，
    # 而False对应一维，expand是改变维度，使其满足 m * n)
    xx = (test_x ** 2).sum(dim=1, keepdim=True).expand(m, n)
    #最后增添了转置操作
    yy = (train_x ** 2).sum(dim=1, keepdim=True).expand(n, m).transpose(0, 1)
    #计算近邻距离公式
    dist_mat = xx + yy - 2 * test_x.matmul(train_x.transpose(0, 1))
    #对距离进行排序
    mink_idxs = dist_mat.argsort(dim=-1)
    #定义一个空列表
    res = []
    for idxs in mink_idxs:
        # voting
        #代码下方会附上解释np.bincount()函数的博客
        res.append(np.bincount(np.array([train_y[idx] for idx in idxs[:k]])).argmax())

    assert len(res) == len(test_y)
    print("acc", accuracy_score(test_y, res))
    #计算运行时长
    time_elapsed = time.time() - since
    print('KNN mat training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return  accuracy_score(test_y, res)
#欧几里得距离计算公式
def cal_distance(x, y):
    return torch.sum((x - y) ** 2) ** 0.5
# KNN的迭代函数

if __name__ == "__main__":
    #加载数据集（下载数据集）
    # train_dataset = datasets.MNIST(root="./data", download= True, transform=transforms.ToTensor(), train=True)
    # test_dataset = datasets.MNIST(root="./data", download= True, transform=transforms.ToTensor(), train=False)
    # 组织训练，测试数据
    train_x = []
    train_y = []
    train_Dataset = MyData(training=0)
    # train_iter = DataLoader(train_Dataset, batch_size=256, shuffle=False)
    test_Dataset = MyData(training=2)
    # test_iter = DataLoader(test_Dataset, batch_size=256, shuffle=False)
    train_features = []
    train_label = []
    test_features = []
    test_label = []

    for att, label in train_Dataset:
        train_features.append(att)
        train_label.append(label)

    for att, label in test_Dataset:
        test_features.append(att)
        test_label.append(label)

    train_features = np.array(train_features)
    train_label = np.array(train_label)
    test_features = np.array(test_features)
    test_label = np.array(test_label)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_features, train_label)
    y_pre = knn.predict(test_features)
    print('Test score is {:.4f}'.format(np.mean(y_pre == test_label)))

    print('accuracy:{}'.format(accuracy_score(test_label, y_pre)))
    print('precision:{}'.format(precision_score(test_label, y_pre, average='micro')))
    print('recall:{}'.format(recall_score(test_label, y_pre, average='micro')))
    print('f1-score:{}'.format(f1_score(test_label, y_pre, average='micro')))
    # acc=KNN(train_features,train_label,test_features,test_label,7)

    # print(acc)
    # for epoch in range(1):
    #     acc=0
    #     for train_x,train_y in train_iter:
    #         for test_x,test_y in test_iter:
    #             acc=+KNN(train_x,train_y,test_x,test_y,7)
    #     # print("acc", acc/len(train_iter.dataset) * 100)

        # for i in range(len(train_iter)):
        #     img, target = train_iter[i]
        #     train_x.append(img.view(-1))
        #     train_y.append(target)
        # #
        # #     if i > 5000:
        # #         break
        # #
        # # # print(set(train_y))
        # #
        # test_x = []
        # test_y = []
        #
        # for i in range(len(test_iter)):
        #     img, target = test_iter[i]
        #     test_x.append(img.view(-1))
        #     test_y.append(target)
        #
        #     if i > 200:
        #         break
        #
        # print("classes:", set(train_y))
    #     KNN(torch.stack(train_x).cuda(), torch.stack(train_y).cuda(), torch.stack(test_x).cuda(), torch.stack(test_y).cuda(), 7)
    #     KNN_by_iter(torch.stack(train_x).cuda(), torch.stack(train_y).cuda(), torch.stack(test_x).cuda(), torch.stack(test_y).cuda(), 7)
    # #
    # test_x,test_y=MyData(training=2)
    #

