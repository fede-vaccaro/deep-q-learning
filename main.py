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
    gamma = 0.90
    e_rate_start = 0.9
    e_rate_end = 0.1

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    opt = torch.optim.Adam(lr=1e-4, params=dqn.parameters())

    dqn.to(device)

    replay_memory = []

    for e in range(n_episodes):
        # reset game!
        # game = GridGame(dim=16)

        for s in range(max_steps_per_episode):
            opt.zero_grad()
            state = torch.tensor(game.state).permute((2, 0, 1)).contiguous()
            x = preprocess(state).to(device).unsqueeze(0)

            state_before = x

            action_net = dqn(x)

            # random action
            act_index = random.randint(0, 3)
            action_rand = torch.zeros(4)
            action_rand[act_index] = 1.0

            current_e_rate = e_rate_start * (1 - e / n_episodes) + e_rate_end * e / n_episodes
            if random.uniform(0.0, 1.0) > current_e_rate:
                action = action_net
                argmax_q = torch.argmax(action_net)
            else:
                action = action_rand
                argmax_q = act_index

            reward = game.action(action.detach().cpu().argmax())

            state = torch.tensor(game.state).permute((2, 0, 1)).contiguous()
            x_after = preprocess(state).to(device).unsqueeze(0)

            with torch.no_grad():
                action_after = dqn(x_after)

            replay_memory += [x, action, x_after]

            if reward != 1:
                y = reward + gamma * action_after.max()
            else:
                y = reward

            loss = (y - action_net.squeeze(0)[argmax_q]) ** 2
            loss.backward()

            opt.step()

            if s % 25 == 0:
                print("Loss at s{}/e{}: {}; current e_rate: {}".format(s + 1, e + 1, loss, current_e_rate))

            if game.is_terminal:
                print("Terminal game!")
                game = GridGame(dim=8)

    # play a game and show how the agent acts!
    game = GridGame(dim=8)
    with torch.no_grad():
        while not game.is_terminal:
            state = torch.tensor(game.state).permute((2, 0, 1)).contiguous()
            x = preprocess(state).to(device).unsqueeze(0)
            action = dqn(x)
            game.action(action.argmax())
            game.visualize_state()


if __name__ == '__main__':
    main()
