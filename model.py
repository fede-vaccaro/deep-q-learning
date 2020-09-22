import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_dim, use_batch_norm):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(3, 3))
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(32)
        conv_1_out = input_dim - 3 + 1

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5))
        if use_batch_norm:
            self.bn2 = nn.BatchNorm2d(64)
        conv_2_out = conv_1_out - 5 + 1
        conv_2_out //= 2  # max pool

        self.dense = nn.Linear(in_features=(conv_2_out ** 2) * 64, out_features=256)
        self.dense2 = nn.Linear(in_features=256, out_features=256)
        self.dense3 = nn.Linear(in_features=256, out_features=4)

        self.use_batch_norm = use_batch_norm

    def forward(self, x):
        y = self.conv1(x)
        if self.use_batch_norm:
            y = self.bn1(y)
        y = F.relu(y)

        y = self.conv2(y)
        if self.use_batch_norm:
            y = self.bn2(y)
        y = F.relu(y)

        y = F.max_pool2d(y, 2)

        # flatten
        batch_size = y.size()[0]
        y = y.view(batch_size, -1)

        y = self.dense(y)
        y = F.relu(y)

        y = self.dense2(y)
        y = F.relu(y)

        y = self.dense3(y)
        q_scores = y

        return q_scores


class ReplayMemory:

    def __init__(self, device, max_dim=50000):
        self.device = device
        self.replay_memory = []
        self.max_dim = max_dim

    def add_sample(self, x, action, x_then, r):
        sample = (x, action, x_then, r)
        if len(self.replay_memory) == self.max_dim:
            random.shuffle(self.replay_memory)
            self.replay_memory = self.replay_memory[1:]

        self.replay_memory += [sample]

    def get_sample(self, minibatch_size=32):
        if self.replay_memory.__len__() > 0:
            mini_batch = random.sample(self.replay_memory, min(len(self.replay_memory), minibatch_size))

            x_batch = torch.cat([x[0] for x in mini_batch], dim=0)
            actions_batch = torch.cat([x[1] for x in mini_batch], dim=0).to(self.device)
            x_then_batch = torch.cat([x[2] for x in mini_batch], dim=0)
            reward_batch = torch.stack([torch.tensor([x[3]], dtype=torch.float32) for x in mini_batch], dim=0).to(
                self.device)

            return x_batch, actions_batch, x_then_batch, reward_batch
        else:
            return None, None, None, None


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

    def add_frame(self, frame):
        self.mem = self.mem[1:]
        self.mem += [frame.clone()]
