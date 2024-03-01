import pandas as pd
import csv
import glob
import numpy as np
import torch.nn as nn
import torch

from sklearn.decomposition import PCA


def make_data():

    files = glob.glob("totaldata/*.csv")
    files[3], files[4] = files[4], files[3]  # DoS:0 Fuzzy:1 gear2 RPM:3 normal_run: 4

    num = 0
    with open("totaldata/carhack/totaldata.csv", "w", newline="") as outfile:
        header = ["timestamp", "id", "dlc", "data", "type"]
        writer = csv.writer(outfile)
        writer.writerow(header)
        for i in range(len(files)):

            print(files[i], i)
            with open(files[i], "r") as csvfile:
                # 创建CSV读取器对象
                reader = csv.reader(csvfile)
                # 遍历每一行数据并进行替换操作
                for row in reader:
                    if len(row) < 12:
                        continue
                    cols_merge = range(3, len(row) - 1)
                    merge = ["".join(map(str, row[i])) for i in cols_merge]
                    merge = "".join(merge)
                    row = row[:3] + [merge] + [row[-1]]

                    # row=row[:-1]+['00']*(12-len(row))+[row[-1]]
                    if row[-1] == 0:

                        row[-1] = 4  # 正常编码
                    elif row[-1] == 1:
                        # num_str=''.join(map(str,code))
                        row[-1] = i
                    # 输出修改后的行数据
                    writer.writerow(row)
    outfile.close()
    # csvfile.close()
    count = 0
    for count, line in enumerate(open("totaldata/totaldata.csv", "r")):
        count += 1
    print(count)  # 17558346
    return


def del_data():
    # 打开原始csv文件和写入新csv文件
    with open("totaldata/carhack/totaldata.csv", "r") as f_input, open(
        "del_data", "w", newline=""
    ) as f_output:

        # 读取原始csv文件
        reader = csv.reader(f_input)

        # 写入新csv文件
        writer = csv.writer(f_output)

        # 写入新csv文件的表头
        writer.writerow(next(reader))

        # 初始化计数器和字典
        counter = 0

        # 逐行遍历原始csv文件，处理符合条件的数据，并写入新csv文件
        for row in reader:
            feature = row[-1]
            if feature in counter and counter < 500000:
                writer.writerow(row)
                counter[feature] += 1

            # 检查计数器是否都达到了100
            if all(value >= 100 for value in counter.values()):
                break


def spilt_data(train_size, val_size):
    data = pd.read_csv("totaldata/carhack/totalid_data.csv")
    print(len(data))
    data = data.drop_duplicates()
    print(len(data))
    # data.to_csv('dropdata.csv',index=False)
    train_data = data.sample(frac=train_size, random_state=0)
    val_data = data.drop(train_data.index).sample(frac=val_size, random_state=1)
    test_data = data.drop(train_data.index).drop(val_data.index)
    print(
        len(train_data), len(val_data), len(test_data)
    )  # 531660 66458 66457 390877 48860 48859
    test_data.to_csv("test_data.csv", index=False)
    val_data.to_csv("val_data.csv", index=False)
    train_data.to_csv("train_data.csv", index=False)

def processing():
    data = pd.read_csv("totaldata/B-CAN/totaldata.csv")

    labelarray = data["type"]
    payload = data["data"]
    payload = [[int(id, 16) for id in hex_id] for hex_id in payload]
    pca = PCA(n_components=3)  # 加载PCA算法，设置降维后主成分数目为2
    data1 = pca.fit_transform(payload)

    print(len(data1) // 30)
    result = []
    for i in range(0, len(data1) // 30):
        sub_arr = []
        label = [4]
        for j in range(30):
            if labelarray[i * 30 + j] < 4:

                label[0] = labelarray[i * 30 + j]

            sub_arr += list(data1[i * 30 + j])
        sub_arr += label
        result.append(sub_arr)
    with open("totaldata/B-CAN/newtotalid_data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        for i in range(len(result)):
            writer.writerow(result[i])
    file.close()


# processing()
class MyData:
    def __init__(self, training=None):
        if training == 0:
            self.data = pd.read_csv("totaldata/carhack/train_data.csv")
        elif training == 1:
            self.data = pd.read_csv("totaldata/carhack/val_data.csv")
        elif training == 2:
            self.data = pd.read_csv("totaldata/carhack/test_data.csv")
        elif training == 3:
            self.data = pd.read_csv(
                "totaldata/M-CAN Intrusion Dataset/1/newtotalid_data.csv"
            )
        elif training == 4:
            self.data = pd.read_csv("totaldata/1_Submission/newtotalid_data.csv")
        elif training == 5:
            self.data = pd.read_csv("totaldata/B-CAN/newtotalid_data.csv")
        self.label = self.data.iloc[:, -1]

        self.data = self.data.iloc[:, :-1]
        self.data = np.array(self.data)
        self.label = np.array(self.label)
        #   self.data=pd.read_csv('my_tensor.csv')
        #   self.label=self.data.iloc[:,-1]
        #   self.data=self.data.iloc[:,:-1]
        #     self.data = os.listdir(data_dir)
        # else:
        #     self.data=os.listdir(data_dir)  #根据参数获得训练集和验证集
        # self.data_path = []
        # for index in range(len(self.data)):
        #     self.data_path.append(os.path.join(data_dir, self.data[index]))
        # self.data = pd.read_csv(self.data_path[index])
        # input_data = self.data.iloc[:, :-1].values
        # self.IDarray = self.data.iloc[:, 1].values
        # self.IDarray = [s.split() for s in self.IDarray]
        # self.IDarray = one_hot(self.IDarray)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # data=torch.tensor(data)
        # data = data.iloc[:,:-1].values
        # IDarray=pd.Series(IDarray)
        # ID1=pd.get_dummies(IDarray[0])
        # data=torch.LongTensor(data)
        # data=torch.tensor(data.values)
        # data=torch.from_numpy(input_data)
        data = self.data[index]

        label = self.label[index]
        data = torch.tensor(data)
        label = torch.tensor(label)
        return data, label
