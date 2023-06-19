import torch.nn as nn
import config
class BP(nn.Module):
    def __init__(self):
        super(BP, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(config.seq_len*config.d_model, config.seq_len),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(config.seq_len,config.d_model),
            nn.ReLU()
        )
        self.fc3=nn.Sequential(
            nn.Linear(config.d_model, 5),

        )
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

