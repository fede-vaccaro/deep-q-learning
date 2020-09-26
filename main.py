import time

import numpy as np
from PIL import Image
from pip._vendor.distlib.compat import raw_input
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from game import GridGame
from model import *
import torch
import torchvision
import random
from test import test
import matplotlib.pyplot as plt


def cat(*args):
    t_list = [*args]
    if t_list[1] != None:
        return torch.cat(t_list, dim=0)
    else:
        return t_list[0]


game_dim = 16
n_episodes = 500
gamma = 0.90
e_rate_start = 0.90
e_rate_end = 0.1
swap_freq = 10

obs_dim = 84  # x 84
use_dql = False
use_batch_norm = True

game_params = {
    'dim': game_dim,
    # 'start': (0, 0),
    'n_holes': 16
}


def main():
    dqn = DQN(input_dim=obs_dim, use_batch_norm=use_batch_norm)
    if use_dql:
        dqn_target = DQN(input_dim=obs_dim, use_batch_norm=use_batch_norm)
    else:
        dqn_target = dqn

    game = GridGame(**game_params)

    dqn.__setattr__('name', 'net')
    dqn_target.__setattr__('name', 'target')

    device = 'cuda'

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    opt = torch.optim.Adam(lr=1e-4, params=dqn.parameters())
    if use_dql:
        target_opt = torch.optim.Adam(lr=1e-4, params=dqn_target.parameters())

    dqn.to(device)
    dqn_target.to(device)

    replay_memory = ReplayMemory(device)

    exploration_stop = 0.5

    lambda1 = lambda e: max(
        e_rate_start * (1 - e / n_episodes * 1 / exploration_stop) + e_rate_end * e / n_episodes * 1 / exploration_stop,
        e_rate_end)
    scheduler = LambdaLR(opt, lr_lambda=[lambda1])

    losses = []
    rewards = []

    train_t0 = time.time()
    for e in range(n_episodes):
        # reset game!
        epoch_reward = []

        # for s in range(max_steps_per_episode):
        epoch_loss = []
        tqdm_ = tqdm(range(1000))
        t = time.time()
        for s in tqdm_:
            # while not game.is_terminal:
            opt.zero_grad()
            state = game.get_state()  # .permute((2, 0, 1))
            x = preprocess(state).to(device).unsqueeze(0)

            dqn.train(False)
            action_net = dqn(x)

            # random action
            act_index = random.randint(0, 3)
            action_rand = torch.zeros(4)
            action_rand[act_index] = 1.0
            action_rand = action_rand.unsqueeze(0).to(device)

            current_e_rate = lambda1(e)
            if random.uniform(0.0, 1.0) > current_e_rate:
                action = action_net
            else:
                action = action_rand

            reward = game.action(action.detach().cpu().argmax())

            state = game.get_state()
            x_after = preprocess(state).to(device).unsqueeze(0)

            replay_memory.add_sample(x, action.argmax(dim=1), x_after,
                                     reward)

            # sample from replay memory
            x_batch, actions_batch, x_then_batch, reward_batch = replay_memory.get_sample(32)

            dqn.train(True)
            Q_predicted = dqn(x_batch)
            with torch.no_grad():
                dqn_target.train(False)
                Q_then_predicted = dqn_target(x_then_batch)

            gt_non_terminal = reward_batch + gamma * Q_then_predicted.max(dim=1)[0]
            gt_terminal = reward_batch
            gt = torch.where(reward_batch != 2, gt_non_terminal, gt_terminal)

            loss = (gt - torch.gather(Q_predicted, 1, actions_batch.unsqueeze(-1))) ** 2
            loss = loss.mean() + dqn.get_reg_loss(1e-5)
            loss.backward()

            epoch_loss += [float(loss)]

            opt.step()

            tqdm_.set_description(
                "Loss at s{}-e{}/{}: {}; current e_rate: {}".format(s + 1, e + 1, n_episodes, float(loss),
                                                                    current_e_rate))

            if game.is_terminal:
                print("Terminal game! Step before ending: {}; Reward: {}".format(game.step_count, game.total_reward))
                epoch_reward.append(game.total_reward)
                game = GridGame(**game_params)
                # break

        if len(epoch_reward) > 0:
            rewards.append(np.array(epoch_reward).mean())
        print("Time for epoch {}:{}s".format(e + 1, int(time.time() - t)))

        epoch_loss = np.array(epoch_loss).mean()
        losses += [epoch_loss]

        if (e + 1) % 100 == 0:
            torch.save({
                'model': dqn.state_dict(),
                'opt': opt.state_dict(),
                'epoch': e,
            }, 'dqn_training_e{}_game_dim{}.ptd'.format(n_episodes, game_dim))

            # scheduler.step()

        if ((e + 1) % swap_freq == 0) and use_dql:
            print("SWAPPING NETWORKS & OPTIMIZERS!")
            dqn, dqn_target = dqn_target, dqn
            opt, target_opt = target_opt, opt

    print("Training time: {} minutes".format((train_t0 - time.time())//60))

    description = "dql" if use_dql else ""

    plt.plot(losses)
    plt.ylabel('loss')
    plt.savefig('loss_per_epoch_{}_{}.pdf'.format(n_episodes, description))
    # plt.show()

    plt.plot(rewards)
    plt.ylabel('rewards')
    plt.savefig('rewards_per_epoch_{}_{}.pdf'.format(n_episodes, description))
    # plt.show()

    if dqn.name == 'target':
        dqn = dqn_target

    # save model for testing
    torch.save(dqn.state_dict(), 'dqn_e{}_game_dim{}_{}.ptd'.format(n_episodes, game_dim, description))

    test(device=device, dqn=dqn, game_params=game_params, obs_dim=obs_dim, preprocess=preprocess, draw_gif=True)


if __name__ == '__main__':
    main()
