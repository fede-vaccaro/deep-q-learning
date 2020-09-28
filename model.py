import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque


class DQN(nn.Module):
    def __init__(self, input_dim, use_batch_norm):
        super(DQN, self).__init__()

        stride1 = 4
        kernel_size1 = 8
        side_in1 = input_dim
        conv_1_out = (side_in1 - kernel_size1) // stride1 + 1

        stride2 = 2
        kernel_size2 = 4
        side_in2 = conv_1_out
        conv_2_out = (side_in2 - kernel_size2) // stride2 + 1

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=kernel_size1, stride=stride1)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size2, stride=stride2)
        if use_batch_norm:
            self.bn2 = nn.BatchNorm2d(32)
        # conv_2_out //= 2  # max pool

        self.dense = nn.Linear(in_features=(conv_2_out ** 2) * 32, out_features=256)
        self.dense2 = nn.Linear(in_features=256, out_features=4)

        self.use_batch_norm = use_batch_norm

    def get_reg_loss(self, lambda_reg):
        reg = torch.tensor([0.0])
        reg = reg.to('cuda')
        for param in self.parameters():
            reg += param.norm(2.0).pow(2.0)

        return reg * lambda_reg

    def forward(self, x):
        y = self.conv1(x)
        if self.use_batch_norm:
            y = self.bn1(y)
        y = F.relu(y)

        y = self.conv2(y)
        if self.use_batch_norm:
            y = self.bn2(y)
        y = F.relu(y)

        # flatten
        batch_size = y.size()[0]
        y = y.view(batch_size, -1)

        y = self.dense(y)
        y = F.relu(y)

        y = self.dense2(y)
        q_scores = y

        return q_scores


class MlpDQN(nn.Module):
    def __init__(self, input_dim, use_batch_norm):
        super(MlpDQN, self).__init__()
        self.use_batch_norm = use_batch_norm

        self.dense = nn.Linear(in_features=input_dim, out_features=1024)

        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(1024)

        self.dense2 = nn.Linear(in_features=1024, out_features=4)

    def get_reg_loss(self, lambda_reg):
        reg = torch.tensor([0.0], requires_grad=True)
        reg = reg.to('cuda')
        for param in self.parameters():
            reg += param.norm(2.0)#.pow(2.0)

        return reg * lambda_reg

    def forward(self, x):
        # flatten
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)

        y = self.dense(x)
        if self.use_batch_norm:
            y = self.bn(y)

        y = F.relu(y)
        y = self.dense2(y)
        q_scores = y

        return q_scores


class ReplayMemory:

    def __init__(self, device, max_dim=100000):
        self.device = device
        self.replay_memory = deque(maxlen=max_dim)
        self.max_dim = max_dim

    def add_sample(self, x, action, x_then, r):
        sample = (x, action, x_then, r)
        # if len(self.replay_memory) == self.max_dim:
        #    random.shuffle(self.replay_memory)
        #    self.replay_memory = self.replay_memory[1:]

        self.replay_memory.append(sample)

    def get_sample(self, minibatch_size=32):
        mini_batch = random.sample(list(self.replay_memory), min(len(self.replay_memory), minibatch_size))

        x_batch = torch.cat([x[0] for x in mini_batch], dim=0)
        actions_batch = torch.cat([x[1] for x in mini_batch], dim=0).to(self.device)
        x_then_batch = torch.cat([x[2] for x in mini_batch], dim=0)
        reward_batch = torch.stack([torch.tensor([x[3]], dtype=torch.float32) for x in mini_batch], dim=0).to(
            self.device)

        return x_batch, actions_batch, x_then_batch, reward_batch


class FrameBuffer:
    def __init__(self, frame_dim, device, mem_length=4):
        self.mem_length = mem_length
        self.device = device
        self.mem = []

        for i in range(mem_length):
            self.mem.append(torch.zeros(1, frame_dim, frame_dim).to(self.device))

    def get_buffer(self):
        fb = torch.cat(self.mem, dim=0).unsqueeze(0)
        return fb.clone()

    def view_buffer(self):
        for state in self.mem:
            s_np = state.cpu().numpy()[0]
            plt.imshow(s_np)
            plt.show()

    def add_frame(self, frame):
        self.mem = self.mem[1:]
        self.mem += [frame.clone()]
