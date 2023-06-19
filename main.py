import pandas as pd
import csv
import glob
import numpy as np
import torch.nn as nn
import torch
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
    files = glob.glob('totaldata/B-CAN/*.csv')
    # files[3],files[4]=files[4],files[3]#DoS:0 Fuzzy:1 gear2 RPM:3 normal_run: 4
    num=[0,1,4]
    j=0
    with open('totaldata/B-CAN/totaldata.csv', 'w', newline='') as outfile:
        header=['timestamp','id','dlc','data','type']
        writer = csv.writer(outfile)
        writer.writerow(header)
        for i in range(len(files)):

            print(files[i],num[i])
            with open(files[i], 'r') as csvfile:
                # 创建CSV读取器对象
                reader = csv.reader(csvfile)
                # print(len(reader))
                next(reader)
                    # 遍历每一行数据并进行替换操作
                for row in reader:
                    # for i in row:
                    #     i=i.split('')
                    #     print(i)
                    # row=[' '.join(map(str,row))]
                    # print(row[0])
                    # print(row[-1])
                    # if len(row)<12:
                    #     continue
                    # cols_merge=range(3,len(row)-1)
                    # merge=[''.join(map(str,row[i])) for i in cols_merge]
                    # merge=''.join(merge)
                    # row=row[:3]+[merge]+[row[-1]]
                        # row=row[:-1]+['00']*(12-len(row))+[row[-1]]

                    if row[-1] == '0.0':

                            row[-1] = 4 #正常编码
                    elif row[-1] == '1.0':
                            # num_str=''.join(map(str,code))
                            row[-1] = num[i]
                            j+=1
                    elif row[-1] == '0':
                            row[-1] = 4
                    s = str(row[1])
                    s = s[5:]
                    row[1] = s

                    row[3]="".join(row[3].split())
                    if len(row[3])<16:
                        continue
                    # if row[-1]=='Normal':
                    #     # if j>5000:
                    #     #     continue
                    #     row[-1]=4
                    #     j+=1
                    #     # continue
                    #
                    # elif row[-1]=='Flooding':
                    #     row[-1]=0
                    # elif row[-1]=='Fuzzing':
                    #     row[-1]=1
                    #     # continue
                    # elif row[-1]=='Spoofing':
                    #     row[-1]=2
                    #     # continue
                    # elif row[-1]=='Replay':
                    #     # row[-1]=2
                    #     continue
                        # 输出修改后的行数据

                    # print(row)
                    writer.writerow(row)
    outfile.close()
    # csvfile.close()
    count = 0
    for count, line in enumerate(open('totaldata/B-CAN/totaldata.csv', "r")):
        count += 1
    print(count,j),#17558346
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
make_data()
def spilt_data(train_size,val_size):
    data=pd.read_csv("totaldata/carhack/newtotalid_data.csv")
    print(len(data))
    data = data.drop_duplicates()
    print(len(data))
    train_data=data.sample(frac=train_size,random_state=0)
    val_data=data.drop(train_data.index).sample(frac=val_size,random_state=1)
    test_data=data.drop(train_data.index).drop(val_data.index)
    print(len(train_data),len(val_data),len(test_data)) #531660 66458 66457 390877 48860 48859
    test_data.to_csv('test_data.csv',index=False)
    val_data.to_csv('val_data.csv',index=False)
    train_data.to_csv('train_data.csv',index=False)

def processing():
    data=pd.read_csv("totaldata/1_Submission/totaldata1.csv",dtype={"id":str})
    # 1. 对 timestamp 特征进行归一化处理
    # timestamp = np.array(data["timestamp"]) # 假设 timestamp 已经被读入为一个 numpy 数组
    # normalized_timestamp = (timestamp - timestamp.min()) / (timestamp.max() - timestamp.min())
    # normalized_timestamp_tensor = torch.tensor(normalized_timestamp, dtype=torch.float32)
    # 2. 对 id 转换为十进制进行归一化
    labelarray=data["type"]
    idarray=data["id"]
    idarray=[[int(id,16)for id in hex_id] for hex_id in idarray]
    # idarray = np.delete(idarray, 0, axis=1)
    idarray=np.array(idarray)
    # print(idarray.shape())

    print(len(idarray)//30)
    result=[]
    for i in range(0,len(idarray)//30):
        sub_arr=[]
        label = [4]
        for j in range(30):
            if labelarray[i*30+j]<4:
                # print(labelarray[i*30+j])
                label[0]= labelarray[i*30+j]
                # print(label[0])
            sub_arr+=list(idarray[i*30+j])
        sub_arr+=label
        # print(label)
        result.append(sub_arr)
    with open('totaldata/1_Submission/totalid_data1.csv','w',newline='') as file:
        writer=csv.writer(file)
        for i in range(len(result)):
            writer.writerow(result[i])
    file.close()
# spilt_data(0.8,0.5)
# processing()
# spilt_data(train_size=0.8,val_size=0.5) #   390877 48860 48859
# pd=pd.read_csv('train_data.csv') #390877 167058
#
# pd=pd.read_csv("totaldata/M-CAN Intrusion Dataset/totalid_data.csv")
# pd=pd.iloc[:,-1]
# data=np.array(pd)
# j=0
# s=0
# m=0
# n=0
# for i in range(len(data)):
#     if data[i]==0:
#         j+=1
#     elif data[i]==1:
#         s+=1
#     elif data[i] == 2:
#         m += 1
#     elif data[i]==3:
#         n+=1
#
#
# print(len(data),j,s,m,n)
# print(len(data))
# data = data.drop_duplicates()
# print(len(data))
