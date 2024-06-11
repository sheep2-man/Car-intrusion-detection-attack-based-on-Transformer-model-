import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

import config
from getdata import MyData


def epoch_time(start_time, end_time):

    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class DNN(nn.Module):
    def __init__(self, dropout):  # 256 20 64  (512,126,32) 90*64
        super(DNN, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(config.seq_len, config.d_model), nn.ReLU(), nn.Dropout(p=dropout)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(config.d_model, 5),
        )

    def forward(self, x):
        # x = x.view(-1, seq_len * d_model)  # 256 20 64
        x = self.fc1(x)
        x = self.fc2(x)
        # x=self.fc3(x)
        # x = self.linear1(x)
        # x = self.linear1_relu(x)
        # x = self.linear1_drop(x)
        # x = self.linear2(x)
        # x = self.linear2_relu(x)
        # x = self.linear2_drop(x)
        # x = self.linear3(x)

        # x = self.linear3_act(x)
        # x = self.linear3_drop(x)
        # x = self.linear4(x)
        return x


class BP(nn.Module):
    def __init__(self):
        super(BP, self).__init__()

        # self.fc1 = nn.Sequential(
        #     nn.Linear(config.seq_len , config.seq_len*2),
        #     nn.ReLU()
        # )

        self.fc2 = nn.Sequential(nn.Linear(config.seq_len, config.d_model), nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(config.d_model, 5),
        )

    def forward(self, x):
        # x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def multiclass_false_positive_rate(y_true, y_pred):
    """
    计算多分类误报率
    """
    fp = 0
    tn = 0
    for i in range(len(y_pred)):
        if y_pred[i] != y_true[i]:

            fp += 1
        else:
            tn += 1
    return fp / (fp + tn)


def main():
    if config.model == "DNN":
        model = DNN(dropout=config.dropout).cuda()
    elif config.model == "BP":
        model = BP().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=config.learing_rate, momentum=config.momentum
    )
    # train_Dataset = MyData(training=0)
    # train_iter = DataLoader(train_Dataset, batch_size=config.trainbatchsize, shuffle=False)
    # for epoch in range(config.epochtime):
    #     model.train()
    #     train_loss = 0
    #     train_acc = 0
    #     label_true, label_pred = [], []
    #     start_time=time.time()
    #     for enc_inputs, dec_outputs in train_iter:
    #         '''
    #         enc_inputs: [batch_size, src_len]
    #         dec_inputs: [batch_size, tgt_len]
    #         dec_outputs: [batch_size, tgt_len]
    #         '''
    #         enc_inputs, dec_outputs = enc_inputs.to(torch.float).cuda(), dec_outputs.cuda()
    #         # outputs: [batch_size * tgt_len, tgt_vocab_size]
    #         outputs = model(enc_inputs)
    #         # outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
    #
    #         loss = criterion(outputs, dec_outputs.view(-1))
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         # print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.10f}'.format(loss))
    #
    #         train_loss += loss
    #         pred = outputs.argmax(dim=1)
    #         train_acc += torch.eq(pred, dec_outputs).sum().cpu().detach().numpy()
    #         label_true.extend(dec_outputs.cpu())
    #         label_pred.extend(pred.cpu())
    #
    #     f1 = f1_score(label_true, label_pred, average='weighted')
    #
    #     end_time = time.time()
    #     epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    #     # print(train_loss)
    #     # print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(train_loss/len(train_iter)),'trainacc =', '{:.6f}'.format(train_acc/len(train_iter.dataset)),f'Time: {epoch_mins}m {epoch_secs}s')
    #     #
    #     print(
    #         "Epoch:%04d : train_loss:%.10f   train_acc:%.10f%%  F1 scor:%.5f time:%dm %0.2fs" % (
    #             epoch + 1, train_loss / len(train_iter), train_acc / len(train_iter.dataset) * 100, f1, epoch_mins,
    #             epoch_secs))
    modelname = (
        "mymodel/model"
        + str(config.trainbatchsize)
        + "-"
        + str(config.epochtime)
        + "-"
        + str(config.learing_rate)
        + str(config.model + str(config.dropout))
    )
    # # # batchsize+epochtime
    # #
    # torch.save(model.state_dict(), modelname + '.pth')
    # torch.save(model, modelname + '.pt')
    #

    model.load_state_dict(torch.load(modelname + ".pth"))
    model = torch.load(modelname + ".pt")
    model.eval()
    test_Dataset = MyData(training=5)
    testiter = DataLoader(test_Dataset, batch_size=256, shuffle=False)
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for epoch in range(1):
            label_true, label_pred = [], []
            for enc_inputs, dec_outputs in testiter:
                """
                enc_inputs: [batch_size, src_len]
                dec_inputs: [batch_size, tgt_len]
                dec_outputs: [batch_size, tgt_len]
                """
                enc_inputs, dec_outputs = (
                    enc_inputs.to(torch.float).cuda(),
                    dec_outputs.cuda(),
                )
                outputs = model(enc_inputs)
                loss = criterion(outputs, dec_outputs.view(-1))
                pred = outputs.argmax(dim=1)
                test_acc += torch.eq(pred, dec_outputs).sum().cpu().detach().numpy()
                # print( 'valoss =', '{:.10f}'.format(loss))
                test_loss += loss
                label_true.extend(dec_outputs.cpu())
                label_pred.extend(pred.cpu())
            precision = precision_score(label_true, label_pred, average="macro")
            recall = recall_score(label_true, label_pred, average="macro")
            f1 = f1_score(label_true, label_pred, average="weighted")
            false = multiclass_false_positive_rate(label_true, label_pred)
            print(
                "Epoch:%04d : test_loss:%.10f   test_acc:%.10f%% f1:%.6f precision:%.6f recall:%.6f false:%.6f"
                % (
                    epoch + 1,
                    test_loss / len(testiter),
                    test_acc / len(testiter.dataset) * 100,
                    f1,
                    precision,
                    recall,
                    false,
                )
            )


main()
