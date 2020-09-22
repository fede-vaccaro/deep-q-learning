import numpy as np
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR

from game import GridGame
from model import DQN, ReplayMemory
import torch
import torchvision
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


def cat(*args):
    t_list = [*args]
    if t_list[1] != None:
        return torch.cat(t_list, dim=0)
    else:
        return t_list[0]


game_dim = 16


def main():
    dqn = DQN(input_dim=game_dim)
    game = GridGame(dim=game_dim)

    device = 'cuda'

    n_episodes = 300
    gamma = 0.90
    e_rate_start = 0.90
    e_rate_end = 0.1

    mean, std = game.get_stats()

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[mean], std=[std]),
    ])

    opt = torch.optim.Adam(lr=1e-4, params=dqn.parameters())

    dqn.to(device)

    replay_memory = ReplayMemory(device)

    lambda1 = lambda e: max(e_rate_start * (1 - e / n_episodes * 1.5) + e_rate_end * e / n_episodes * 1.5,
                            e_rate_end)
    scheduler = LambdaLR(opt, lr_lambda=[lambda1])
    losses = []

    for e in range(n_episodes):
        # reset game!

        # for s in range(max_steps_per_episode):
        epoch_loss = []
        for s in range(1000):
            # while not game.is_terminal:
            opt.zero_grad()
            state = torch.tensor(game.get_state()).contiguous()  # .permute((2, 0, 1))
            x = preprocess(state).to(device).unsqueeze(0)

            action_net = dqn(x)

            # random action
            act_index = random.randint(0, 3)
            action_rand = torch.zeros(4)
            action_rand[act_index] = 1.0
            action_rand = action_rand.unsqueeze(0).to(device)

            current_e_rate = max(e_rate_start * (1 - e / n_episodes * 1.5) + e_rate_end * e / n_episodes * 1.5,
                                 e_rate_end)
            if random.uniform(0.0, 1.0) > current_e_rate:
                action = action_net
            else:
                action = action_rand

            reward = game.action(action.detach().cpu().argmax())

            state = torch.tensor(game.get_state()).contiguous()
            x_after = preprocess(state).to(device).unsqueeze(0)

            replay_memory.add_sample(x, action.argmax(dim=1), x_after, reward)

            # sample from replay memory
            x_batch, actions_batch, x_then_batch, reward_batch = replay_memory.get_sample(32)

            Q_predicted = dqn(x_batch)
            with torch.no_grad():
                Q_then_predicted = dqn(x_then_batch)

            gt_non_terminal = reward_batch + gamma * Q_then_predicted.max(dim=1)[0]
            gt_terminal = reward_batch
            gt = torch.where(reward_batch < 0, gt_non_terminal, gt_terminal)

            loss = (gt - torch.gather(Q_predicted, 1, actions_batch.unsqueeze(-1))) ** 2
            loss = loss.mean()
            loss.backward()

            epoch_loss += [float(loss)]

            opt.step()

            if (s + 1) % 100 == 0:
                print(
                    "Loss at s{}-e{}/{}: {}; current e_rate: {}".format(s + 1, e + 1, n_episodes, loss, current_e_rate))

            s += 1
            if game.is_terminal:
                print("Terminal game!")
                print("Step per epoch:", s)
                game = GridGame(dim=game_dim)
                break

        epoch_loss = np.array(epoch_loss).mean()
        losses += [epoch_loss]

        # scheduler.step()

    plt.plot(losses)
    plt.ylabel('loss')
    plt.savefig('training_{}.pdf'.format(n_episodes))
    plt.show()

    # play a game and show how the agent acts!
    game = GridGame(dim=game_dim)

    states = []
    max_steps = 1000
    with torch.no_grad():
        for i in range(max_steps):
            state = torch.tensor(game.get_state()).contiguous()
            x = preprocess(state).to(device).unsqueeze(0)
            if random.uniform(0.0, 1.0) < e_rate_end / 2.0:
                action = random.randint(0, 3)
            else:
                action = dqn(x).argmax()

            game.action(action)
            state = Image.fromarray((game.get_state(rgb=True) * 255.0).astype('uint8'), 'RGB').resize((400, 400))
            states.append(state)
            if game.is_terminal:
                print("Agent won in {} steps!".format(i))
                break

    states[0].save('match_{}.gif'.format(n_episodes),
                   save_all=True, append_images=states[1:], optimize=False, duration=150, loop=0)


if __name__ == '__main__':
    main()
