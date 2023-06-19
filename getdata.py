import pandas as pd
import csv
import glob
import numpy as np
import torch.nn as nn
import torch

from sklearn.decomposition import PCA
def make_data():
    # with open('totaldata/normal_run_data.txt','r') as textfile,open('totaldata/normal_run.csv','w',newline='') as csvfile:
    #     # lines=textfile.readline()
    #     # df=pd.DataFrame([x.strip().split() for x in lines])
    #     # cols=df.+list(df.columns[7:-1])
    #     # cols.to_csv('data.csv',index=False)
    #     writer=csv.writer(csvfile)
    #     for row in textfile:
    #         data=row.strip().split()
    #         # tags=data[0].split(';')
    #         # text=data[1]
    #         # print(data[0])
    #         # datastr=','.join(data[7:-1])
    #         # print(datastr)
    #         writer.writerow([data[1],data[3],data[6],*data[7:],'R'])#贴上正常标签
    #     textfile.close()
    #     csvfile.close()
    # #读取正常数据写入data.csv
    # count = -1
    # for count, line in enumerate(open('totaldata/normal_run.csv', "r")):
    #     count += 1
    # print(count)
    files = glob.glob('totaldata/*.csv')
    files[3],files[4]=files[4],files[3]#DoS:0 Fuzzy:1 gear2 RPM:3 normal_run: 4

    num=0
    with open('totaldata/carhack/totaldata.csv', 'w', newline='') as outfile:
        header=['timestamp','id','dlc','data','type']
        writer = csv.writer(outfile)
        writer.writerow(header)
        for i in range(len(files)):
            # df = pd.read_csv(file, header=None, na_values=['NA'])
            # df.fillna('00', inplace=True)
            # df.to_csv('td1.csv', index=False, header=False)
            # code=[0]*len(files)
            # code[i]=1
            print(files[i],i)
            with open(files[i], 'r') as csvfile:
                # 创建CSV读取器对象
                reader = csv.reader(csvfile)
                    # 遍历每一行数据并进行替换操作
                for row in reader:
                    if len(row)<12:
                        continue
                    cols_merge=range(3,len(row)-1)
                    merge=[''.join(map(str,row[i])) for i in cols_merge]
                    merge=''.join(merge)
                    row=row[:3]+[merge]+[row[-1]]

                        # row=row[:-1]+['00']*(12-len(row))+[row[-1]]
                    if row[-1] == 0:

                            row[-1] = 4 #正常编码
                    elif row[-1] == 1:
                            # num_str=''.join(map(str,code))
                            row[-1] = i
                        # 输出修改后的行数据
                    writer.writerow(row)
    outfile.close()
    # csvfile.close()
    count = 0
    for count, line in enumerate(open('totaldata/totaldata.csv', "r")):
        count += 1
    print(count)#17558346
    return
    #对正常数据异常数据进行01和10编码

    # df=pd.read_csv('td.csv',header=None,)
    # data = os.listdir(data_dir)
    # data_path = []
    # for index in range(len(data)):
    #     data_path.append(os.path.join(data_dir, data[index]))
    # data = pd.read_csv(data_path[index])
    #
    #
    #
    # print(data)
    #
    # IDarray = data.iloc[:, 1].values
    # IDarray = [s.split() for s in IDarray]
    # # print(IDarray)
    # # data = one_hot(IDarray)

def del_data():
    # 打开原始csv文件和写入新csv文件
    with open('totaldata/carhack/totaldata.csv', 'r') as f_input, open('del_data', 'w', newline='') as f_output:

        # 读取原始csv文件
        reader = csv.reader(f_input)

        # 写入新csv文件
        writer = csv.writer(f_output)

        # 写入新csv文件的表头
        writer.writerow(next(reader))

        # 初始化计数器和字典
        counter= 0

        # 逐行遍历原始csv文件，处理符合条件的数据，并写入新csv文件
        for row in reader:
            feature = row[-1]
            if feature in counter and counter <500000 :
                writer.writerow(row)
                counter[feature] += 1

            # 检查计数器是否都达到了100
            if all(value >= 100 for value in counter.values()):
                break
def spilt_data(train_size,val_size):
    data=pd.read_csv("totaldata/carhack/totalid_data.csv")
    print(len(data))
    data = data.drop_duplicates()
    print(len(data))
    # data.to_csv('dropdata.csv',index=False)
    train_data=data.sample(frac=train_size,random_state=0)
    val_data=data.drop(train_data.index).sample(frac=val_size,random_state=1)
    test_data=data.drop(train_data.index).drop(val_data.index)
    print(len(train_data),len(val_data),len(test_data)) #531660 66458 66457 390877 48860 48859
    test_data.to_csv('test_data.csv',index=False)
    val_data.to_csv('val_data.csv',index=False)
    train_data.to_csv('train_data.csv',index=False)
    # data_train, data_test, label_train, label_test = train_test_split(data[:,-1], data[-1], test_size=test_size, random_state=42)
    # print(len(data_train),label_train)
# def processing(data):
#     # data=pd.read_csv("totaldata.csv")
#     # 1. 对 timestamp 特征进行归一化处理
#     # timestamp = np.array(data["timestamp"]) # 假设 timestamp 已经被读入为一个 numpy 数组
#     # normalized_timestamp = (timestamp - timestamp.min()) / (timestamp.max() - timestamp.min())
#     # normalized_timestamp_tensor = torch.tensor(normalized_timestamp, dtype=torch.float32)
#     # 2. 对 id 转换为十进制进行归一化
#
#     idarray=data["id"]
#
#     idarray=[[int(id,16)for id in hex_id] for hex_id in idarray]
#
#     idarray = np.delete(idarray, 0, axis=1)
#     idarray=np.array(idarray)
#     # normalized_id=(idarray-idarray.min())/(idarray.max()-idarray.min())
#     normalized_id_tensor=torch.tensor(idarray,dtype=torch.int64)
#
#     # dlc=torch.tensor(np.array(data["dlc"]))
#     # dlc_onehot = F.one_hot(dlc, num_classes=16).float()  # 假设 dlc 已经被读入为一个 numpy 数组
#     # print(dlc_onehot)
#     # id_onehot = F.one_hot(id, num_classes=256).float()  # 假设 id 已经被读入为一个 numpy 数组
#
#
#     # 3. 对 data 标准化操作
#     datarray=data["data"]
#     datarray=[[int(id,16) for id in hex_id ] for hex_id in datarray]
#     # datarray=np.delete(datarray,-1,axis=1)
#     datarray=np.array(datarray)
#     # scaler = StandardScaler()
#     # normalized_datarray = scaler.fit_transform(datarray)#使其满足正态分布
#     # normalized_datarray=(datarray-np.mean(datarray,axis=0))/np.std(datarray,axis=0)
#     # print(normalized_datarray)
#     normalized_datarray_tensor = torch.tensor(datarray, dtype=torch.int64)
#
#
#     # print(scaler.mean_,scaler.var_)#均值方差
#     # print(datarray)
#     # embedding_layer = nn.Embedding(num_embeddings=256, embedding_dim=128)  # 假设使用了一个 Embedding 层来将字节转换为实数向量
#     # embedded_data = embedding_layer(torch.tensor(datarray, dtype=torch.long)).mean(
#     #     dim=1)  # 假设 data 已经被读入为一个 numpy 数组
#     # print(embedded_data)
#
#     # # 4. 将经过处理后的各个特征拼接成一个大的输入向量 timestamp-id-data 1+1+4 六位
#
#
#     # start_token = (-1)*size
#     # start_token = np.array(start_token)
#     # start_token = torch.tensor(start_token,dtype=torch.float32)
#
#     # print(len(start_token))
#     # input_vec = torch.stack((start_token, input_vec), dim=1)
#
#     input_vec=torch.cat([normalized_id_tensor,normalized_datarray_tensor],dim=1)
#     # input_vec = np.insert(input_vec, 0, 17, axis=1)  # 添加开始标记
#     input_vec=np.insert(input_vec,len(input_vec[0]),16,axis=1)# 添加结束标记
#     # print(input_vec,len(input_vec[0])) #19位
#     # input_vec = torch.cat([normalized_timestamp_tensor, normalized_id_tensor], dim=1)#timestamp+id+data
#
#     # label=pd.DataFrame(data,columns=['type'])
#     # print(label)
#     #
#     label=np.array(data["type"])#攻击类型
#     label = torch.tensor(label)
#
#     #
#     # input_vec=input_vec.numpy()
#     # np.savetxt("input",input_vec)
#     # # data=torch.cat([input_vec,label],dim=1)
#     # print(data)
#     # label_onehot = pd.get_dummies(label)
#     # label_onehot=torch.nn.functional.one_hot(label)
#     # label_onehot=np.insert(label_onehot,0,2,axis=1)
#     # label_onehot = np.insert(label_onehot,len(label_onehot[0]), 3, axis=1)
#     #添加开始标志2 结束标志3
#
#     # labelarray=[[a[label]=1] for label in labelarray]
#     # torch.set_printoptions(threshold=np.inf)
#     # print(label_onehot)
#       # tokenizer 是用于添加起始和结束标记的工具
#
#     # start_token = [char_to_idx["[CLS]"]]
#     # end_token = [char_to_idx["[SEP]"]]
#     # input_vec = torch.cat([start_token, input_vec, end_token], dim=1)
#
#     # # 5. 对于长度不足的输入，在末尾填充特殊的填充标记
#     # max_seq_length = 100  # 假设最大序列长度为 100
#     # padding_token = tokenizer.pad_token_id
#     # if input_vec.shape[1] < max_seq_length:
#     #     padding = torch.ones((1, max_seq_length - input_vec.shape[1]), dtype=torch.long) * padding_token
#     #     input_vec = torch.cat([input_vec, padding], dim=1)
#     #
#     # # 6. 将处理后的输入向量传递给 Transformer 模型进行训练或预测
#     # # model = MyTransformerModel()
#     # # output = model(input_vec)
#     # np.savetxt('my_tensor.csv',input_vec,delimiter=',')
#     return input_vec,label
def processing():
    data=pd.read_csv("totaldata/B-CAN/totaldata.csv")
    # 1. 对 timestamp 特征进行归一化处理
    # timestamp = np.array(data["timestamp"]) # 假设 timestamp 已经被读入为一个 numpy 数组
    # normalized_timestamp = (timestamp - timestamp.min()) / (timestamp.max() - timestamp.min())
    # normalized_timestamp_tensor = torch.tensor(normalized_timestamp, dtype=torch.float32)
    # 2. 对 id 转换为十进制进行归一化
    labelarray=data["type"]
    idarray=data["id"]
    payload=data["data"]
    payload = [[int(id, 16) for id in hex_id] for hex_id in payload]
    pca = PCA(n_components=3)  # 加载PCA算法，设置降维后主成分数目为2
    data1 = pca.fit_transform(payload)
    # print(data1)
    # for i in range(len(labelarray)):
    #     if labelarray[i]==1:
    #         pca = PCA(n_components=3)  # 加载PCA算法，设置降维后主成分数目为2
    #         data1 = pca.fit_transform(payload[i])
    #         print(data1)
    #         # idarray[i]=payload

    # idarray=[[int(id,16)for id in hex_id] for hex_id in idarray]
    # idarray = np.delete(idarray, 0, axis=1)
    # idarray=np.array(idarray)
    # print(idarray.shape())
    print(len(data1)//30)
    result=[]
    for i in range(0,len(data1)//30):
        sub_arr=[]
        label = [4]
        for j in range(30):
            if labelarray[i*30+j]<4:
                # print(labelarray[i*30+j])
                label[0]=labelarray[i*30+j]
                # label[0] = 1
            sub_arr+=list(data1[i*30+j])
        sub_arr+=label
        # print(label)
        result.append(sub_arr)
    with open('totaldata/B-CAN/newtotalid_data.csv', 'w', newline='') as file:
        writer=csv.writer(file)
        for i in range(len(result)):
            writer.writerow(result[i])
    file.close()
# processing()
class MyData():
    def __init__(self, training=None):
        if training==0:
            self.data=pd.read_csv('totaldata/carhack/train_data.csv')
        elif training==1:
            self.data = pd.read_csv('totaldata/carhack/val_data.csv')
        elif training==2:
            self.data = pd.read_csv('totaldata/carhack/test_data.csv')
        elif training==3:
            self.data=pd.read_csv('totaldata/M-CAN Intrusion Dataset/1/newtotalid_data.csv')
        elif training == 4:
            self.data = pd.read_csv('totaldata/1_Submission/newtotalid_data.csv')
        elif training==5:
            self.data=pd.read_csv('totaldata/B-CAN/newtotalid_data.csv')
        self.label = self.data.iloc[:, -1]

        self.data = self.data.iloc[:, :-1]
        self.data=np.array(self.data)
        self.label=np.array(self.label)
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

        label=self.label[index]
        data = torch.tensor(data)
        label=torch.tensor(label)
        return data,label

