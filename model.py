import numpy as np
import config
from transformer import Transformer
from DNN import DNN
from BP import BP
from draw import draw
import torch.optim as optim
import torch
from getdata import MyData
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from config import *
import os
import time

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class TDmodel(nn.Module):
    def __init__(self):
        super(TDmodel, self).__init__()
        self.encoder = Transformer().cuda()

        self.dnn = DNN(dropout=dropout).cuda()

    def forward(self, enc_inputs):
        x = self.encoder(enc_inputs)
        x = x.view(-1, seq_len * d_model)  # 256 20 64
        x = self.dnn(x)
        return x


class TPmodel(nn.Module):
    def __init__(self):
        super(TPmodel, self).__init__()
        self.encoder = Transformer().cuda()

        self.BP = BP().cuda()

    def forward(self, enc_inputs):
        x = self.encoder(enc_inputs)
        x = x.view(-1, seq_len * d_model)  # 256 20 64
        x = self.BP(x)
        return x


#
if config.model == "TD":
    model = TDmodel().cuda()
elif config.model == "TP":
    model = TPmodel().cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learing_rate, momentum=momentum)

# modelname='mymodel2/model'+str(config.trainbatchsize)+'-'+str(config.epochtime)+'-'+str(config.learing_rate)+str(config.model+str(config.dropout))
# # # #batchsize+epochtime
# # model.load_state_dict(torch.load(modelname+'.pth'))
# # print(model)
# model=torch.load(modelname+'.pt')


def epoch_time(start_time, end_time):

    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model):
    startime = time.time()
    train_Dataset = MyData(training=0)
    endtime = time.time()
    data_mins, data_secs = epoch_time(startime, endtime)
    print("processingtime:%dm %0.2fs" % (data_mins, data_secs))

    train_iter = DataLoader(train_Dataset, batch_size=trainbatchsize, shuffle=False)
    losslist = []
    acclist = []
    for epoch in range(epochtime):
        model.train()
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        label_true, label_pred = [], []
        for enc_inputs, dec_outputs in train_iter:
            """
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]
            dec_outputs: [batch_size, tgt_len]
            """
            enc_inputs, dec_outputs = enc_inputs.cuda(), dec_outputs.cuda()
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            outputs = model(enc_inputs)
            # outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)

            loss = criterion(outputs, dec_outputs.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.10f}'.format(loss))

            train_loss += loss
            pred = outputs.argmax(dim=1)
            train_acc += torch.eq(pred, dec_outputs).sum().cpu().detach().numpy()
            label_true.extend(dec_outputs.cpu())
            label_pred.extend(pred.cpu())

        f1 = f1_score(label_true, label_pred, average="weighted")

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # print(train_loss)
        # print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(train_loss/len(train_iter)),'trainacc =', '{:.6f}'.format(train_acc/len(train_iter.dataset)),f'Time: {epoch_mins}m {epoch_secs}s')
        #
        print(
            "Epoch:%04d : train_loss:%.10f   train_acc:%.10f%%  F1 scor:%.5f time:%dm %0.2fs"
            % (
                epoch + 1,
                train_loss / len(train_iter),
                train_acc / len(train_iter.dataset) * 100,
                f1,
                epoch_mins,
                epoch_secs,
            )
        )
        loss1 = train_loss.item()
        acc1 = train_acc.item()
        losslist.append(loss1 / len(train_iter))
        acclist.append(acc1 / len(train_iter.dataset) * 100)
    # draw(epoch=epochtime,loss=losslist,acc=acclist,path=str(config.trainbatchsize)+'-'+str(config.epochtime)+'-'+str(config.learing_rate)+str(config.model)+str(config.dropout))


modelname = (
    "mymodel/model"
    + str(config.trainbatchsize)
    + "-"
    + str(config.epochtime)
    + "-"
    + str(config.learing_rate)
    + str(config.model + str(config.dropout))
)


def multiclass_false_positive_rate(y_true, y_pred):
    """
    计算多分类误报率
    """
    fp = 0
    tn = 0
    for i in range(len(y_pred)):
        if y_pred[i] != y_true[i]:
            print(y_pred[i], y_true[i])
            fp += 1
        else:
            tn += 1
    return fp / (fp + tn)


# def val(model):
#     model.eval()
#     val_Dataset = MyData(training=1)
#     valiter = DataLoader(val_Dataset, batch_size=trainbatchsize, shuffle=False)
#     with torch.no_grad():
#         losslist = []
#         acclist = []
#         for epoch in range(1):
#             val_loss = 0
#             val_acc=0
#             label_true,label_pred=[],[]
#             for enc_inputs,  dec_outputs in valiter:
#                 '''
#                 enc_inputs: [batch_size, src_len]
#                 dec_inputs: [batch_size, tgt_len]
#                 dec_outputs: [batch_size, tgt_len]
#                 '''
#                 enc_inputs,  dec_outputs = enc_inputs.cuda(),  dec_outputs.cuda()
#                 outputs = model(enc_inputs)
#                 loss = criterion(outputs, dec_outputs.view(-1))
#
#                 val_loss += loss
#                 pred = outputs.argmax(dim=1)
#                 val_acc += torch.eq(pred, dec_outputs).sum().cpu().detach().numpy()
#                 label_true.extend(dec_outputs.cpu())
#                 label_pred.extend(pred.cpu())
#
#             f1 = f1_score(label_true, label_pred, average='weighted')
#             losslist.append(val_loss/ len(valiter))
#             acclist.append(val_acc / len(valiter.dataset) * 100)
#             print(
#                 "Epoch:%04d : val_loss:%.10f   val_acc:%.10f%%  " % (
#                     epoch + 1, val_loss/ len(valiter) , val_acc / len(valiter.dataset) * 100))
#             # loss1 = val_loss.item()
#             # acc1 = val_acc.item()
#             # losslist.append(loss1 / len(valiter))
#             # acclist.append(acc1 / len(valiter.dataset) * 100)
#         # draw(epoch=epochtime,loss=losslist,acc=acclist)
def val(model):
    model.eval()
    val_Dataset = MyData(training=1)
    valiter = DataLoader(val_Dataset, batch_size=256, shuffle=False)
    test_loss = 0
    test_acc = 0
    # losslist=[]
    # acclist=[]
    with torch.no_grad():
        label_true, label_pred = [], []
        for epoch in range(1):

            for enc_inputs, dec_outputs in valiter:
                """
                enc_inputs: [batch_size, src_len]
                dec_inputs: [batch_size, tgt_len]
                dec_outputs: [batch_size, tgt_len]
                """
                enc_inputs, dec_outputs = enc_inputs.cuda(), dec_outputs.cuda()
                outputs = model(enc_inputs)
                loss = criterion(outputs, dec_outputs.view(-1))
                pred = outputs.argmax(dim=1)
                test_acc += torch.eq(pred, dec_outputs).sum().cpu().detach().numpy()
                # print( 'valoss =', '{:.10f}'.format(loss))
                test_loss += loss
                label_true.extend(dec_outputs.cpu())
                label_pred.extend(pred.cpu())
            precision = precision_score(label_true, label_pred, average="weighted")
            recall = recall_score(label_true, label_pred, average="weighted")
            f1 = f1_score(label_true, label_pred, average="weighted")
            # confusion = confusion_matrix(label_true, label_pred)
            # false_positive_rate = confusion[:, 0].sum() / confusion.sum(axis=1)[0]
            false_positive_rate = multiclass_false_positive_rate(label_true, label_pred)

            print(
                "Epoch:%04d : test_loss:%.10f   test_acc:%.10f%% f1:%.6f precision:%.6f recall:%.6f  false rate:%.6f"
                % (
                    epoch + 1,
                    test_loss / len(valiter),
                    test_acc / len(valiter.dataset) * 100,
                    f1,
                    precision,
                    recall,
                    false_positive_rate,
                )
            )
            # loss1 =test_loss.item()
            # acc1 = test_acc.item()
            # losslist.append(loss1 / len(testiter))
            # acclist.append(acc1 / len(testiter.dataset) * 100)
    # draw(epoch=epochtime, loss=losslist, acc=acclist)


def test(model):
    model.eval()
    test_Dataset = MyData(training=3)
    testiter = DataLoader(test_Dataset, batch_size=256, shuffle=False)
    test_loss = 0
    test_acc = 0
    # losslist=[]
    # acclist=[]
    with torch.no_grad():
        label_true, label_pred = [], []
        for epoch in range(1):
            for enc_inputs, dec_outputs in testiter:
                """
                enc_inputs: [batch_size, src_len]
                dec_inputs: [batch_size, tgt_len]
                dec_outputs: [batch_size, tgt_len]
                """
                enc_inputs, dec_outputs = enc_inputs.cuda(), dec_outputs.cuda()
                outputs = model(enc_inputs)
                loss = criterion(outputs, dec_outputs.view(-1))
                pred = outputs.argmax(dim=1)
                test_acc += torch.eq(pred, dec_outputs).sum().cpu().detach().numpy()
                # print( 'valoss =', '{:.10f}'.format(loss))
                test_loss += loss
                label_true.extend(dec_outputs.cpu())
                label_pred.extend(pred.cpu())
            precision = precision_score(label_true, label_pred, average="weighted")
            recall = recall_score(label_true, label_pred, average="weighted")
            f1 = f1_score(label_true, label_pred, average="weighted")
            # confusion = confusion_matrix(label_true, label_pred)
            # false_positive_rate = confusion[:, 0].sum() / confusion.sum(axis=1)[0]
            false_positive_rate = multiclass_false_positive_rate(label_true, label_pred)

            print(
                "Epoch:%04d : test_loss:%.10f   test_acc:%.10f%% f1:%.6f precision:%.6f recall:%.6f  false rate:%.6f"
                % (
                    epoch + 1,
                    test_loss / len(testiter),
                    test_acc / len(testiter.dataset) * 100,
                    f1,
                    precision,
                    recall,
                    false_positive_rate,
                )
            )
            # loss1 =test_loss.item()
            # acc1 = test_acc.item()
            # losslist.append(loss1 / len(testiter))
            # acclist.append(acc1 / len(testiter.dataset) * 100)
    # draw(epoch=epochtime, loss=losslist, acc=acclist)


# model.load_state_dict(torch.load(modelname+'.pth'))
# print(model)
model = torch.load(modelname + ".pt")
# val(model)
test(model)
