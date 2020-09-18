from PIL import Image

from game import GridGame
from model import DQN
import torch
import torchvision
import random
import numpy as np


def main():
    game = GridGame(dim=8)
    dqn = DQN(input_dim=8)

    device = 'cuda'

    n_episodes = 500
    max_steps_per_episode = game.max_steps ** 2
    gamma = 0.95
    e_rate_start = 0.90
    e_rate_end = 0.1

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    opt = torch.optim.Adam(lr=1e-4, params=dqn.parameters())

    dqn.to(device)

    replay_memory = []
    replay_memory_terminal = []

    for e in range(n_episodes):
        # reset game!
        game = GridGame(dim=8)

        # for s in range(max_steps_per_episode):
        s = 0
        while not game.is_terminal:
            opt.zero_grad()
            state = torch.tensor(game.state).permute((2, 0, 1)).contiguous()
            x = preprocess(state).to(device).unsqueeze(0)

            with torch.no_grad():
                action_net = dqn(x)

            # random action
            act_index = random.randint(0, 3)
            action_rand = torch.zeros(4)
            action_rand[act_index] = 1.0

            current_e_rate = e_rate_start * (1 - e / n_episodes) + e_rate_end * e / n_episodes
            if random.uniform(0.0, 1.0) > current_e_rate:
                action = action_net
            else:
                action = action_rand

            reward = game.action(action.detach().cpu().argmax())

            state = torch.tensor(game.state).permute((2, 0, 1)).contiguous()
            x_after = preprocess(state).to(device).unsqueeze(0)
            with torch.no_grad():
                action_after = dqn(x_after)

            replay_memory += [(x, action, x_after, reward)]

            # sample from replay memory

            mini_batch = random.sample(replay_memory, min(len(replay_memory), 31))
            x_batch = torch.cat([x[0] for x in mini_batch] + [x], dim=0)
            x_then_batch = torch.cat([x[2] for x in mini_batch] + [x_after], dim=0)
            reward_batch = torch.stack([torch.tensor([x[3]], dtype=torch.float32) for x in mini_batch] + [
                torch.tensor([reward], dtype=torch.float32)], dim=0).to(device)

            Q_predicted = dqn(x_batch)
            Q_then_predicted = dqn(x_then_batch)
            gt_non_terminal = reward_batch + gamma * Q_then_predicted.max(dim=1)[0]
            gt_terminal = reward_batch
            gt = torch.where(reward_batch < 0, gt_non_terminal, gt_terminal)

            loss = (gt - Q_predicted.max(dim=1)[0]) ** 2
            loss = loss.mean()
            loss.backward()

            opt.step()

            if s % 25 == 0:
                print("Loss at s{}/e{}: {}; current e_rate: {}".format(s + 1, e + 1, loss, current_e_rate))

            s += 1
            if game.is_terminal:
                print("Terminal game!")

    # play a game and show how the agent acts!
    game = GridGame(dim=8)
    states = []
    max_steps = 1000
    with torch.no_grad():
        for i in range(max_steps):
            state = torch.tensor(game.state).permute((2, 0, 1)).contiguous()
            x = preprocess(state).to(device).unsqueeze(0)
            if random.uniform(0.0, 1.0) < 0.1:
                action = random.randint(0, 3)
            else:
                action = dqn(x).argmax()

            game.action(action)
            state = Image.fromarray((game.state * 255.0).astype('uint8'), 'RGB').resize((400, 400))
            states.append(state)
            if game.is_terminal:
                break

    states[0].save('match.gif',
                   save_all=True, append_images=states[1:], optimize=False, duration=40, loop=0)


if __name__ == '__main__':
    main()
