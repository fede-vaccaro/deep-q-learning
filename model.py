import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3))
        conv_1_out = input_dim - 3 + 1
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(6, 6))
        conv_2_out = conv_1_out - 6 + 1
        self.dense = nn.Linear(in_features=(conv_2_out**2)*16, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=4)

    def forward(self, x):
        y = self.conv1(x)
        y = F.relu(y)

        y = self.conv2(y)
        y = F.relu(y)

        # flatten
        batch_size = y.size()[0]
        y = y.view(batch_size, -1)

        y = self.dense(y)
        y = F.relu(y)

        y = self.dense2(y)
        q_scores = y

        return q_scores
