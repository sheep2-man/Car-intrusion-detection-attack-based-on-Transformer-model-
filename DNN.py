import torch.nn as nn

import config

class DNN(nn.Module):
    def __init__(self,dropout):#256 20 64  (512,126,32) 90*64
        super(DNN, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(config.seq_len * config.d_model, config.seq_len),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(config.seq_len, config.d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(config.d_model, config.numclass),
        )

        # self.linear1 = nn.Linear(config.seq_len*config.d_model, config.seq_len)
        # self.linear1_act = nn.ReLU()
        # self.linear1_drop = nn.Dropout(p=dropout)
        # self.linear2 = nn.Linear(config.seq_len,config.d_model)
        # self.linear2_act = nn.ReLU()
        # self.linear2_drop = nn.Dropout()
        # self.linear3 = nn.Linear(config.d_model, 5)

        # self.linear3_act = nn.ReLU()
        # self.linear3_drop = nn.Dropout()
        # self.linear4 = nn.Linear(16, output_size)

    def forward(self, x):
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        #
        # x = self.linear1(x)
        # x = self.linear1_act(x)
        # x = self.linear1_drop(x)
        # x = self.linear2(x)
        # x = self.linear2_act(x)
        # x = self.linear2_drop(x)
        # x = self.linear3(x)

        # x = self.linear3_act(x)
        # x = self.linear3_drop(x)
        # x = self.linear4(x)
        return x



