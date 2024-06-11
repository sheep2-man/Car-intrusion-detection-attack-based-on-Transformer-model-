from sklearn import tree

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    accuracy_score,
    recall_score,
    f1_score,
)
from getdata import MyData

# def loadDataSet():
#     iris_dataset = load_iris()
#     X = iris_dataset.data
#     y = iris_dataset.target
#     # 将数据划分为训练集和测试集
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     return X_train, X_test, y_train, y_test
# 训练决策树模性
def trainDT(x_train, y_train):
    # DT生成和训练
    clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=10)
    clf.fit(x_train, y_train)
    return clf


# 测试模型
def test(model, x_test, y_test):
    # 将标签转换为one-hot形式
    # y_one_hot = label_binarize(y_test, np.arange(5))
    # 预测结果
    y_pre = model.predict(x_test)
    # 预测结果的概率
    y_pre_pro = model.predict_proba(x_test)

    # 混淆矩阵
    con_matrix = confusion_matrix(y_test, y_pre)
    # print('confusion_matrix:\n', con_matrix)
    print("accuracy:{}".format(accuracy_score(y_test, y_pre)))
    print("precision:{}".format(precision_score(y_test, y_pre, average="micro")))
    print("recall:{}".format(recall_score(y_test, y_pre, average="micro")))
    print("f1-score:{}".format(f1_score(y_test, y_pre, average="micro")))

    # 绘制ROC曲线
    # drawROC(y_one_hot, y_pre_pro)


if __name__ == "__main__":
    train_set = MyData(training=0)
    test_set = MyData(training=2)
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

    model = trainDT(x_train=train_features, y_train=train_label)
    test(model, x_test=test_features, y_test=test_label)
