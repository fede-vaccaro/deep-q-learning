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
n_episodes = 300
gamma = 0.99
e_rate_start = 0.90
e_rate_end = 0.1
swap_freq = 10

obs_dim = 84  # x 84
use_dql = True
use_batch_norm = False

game_params = {
    'dim': game_dim,
    # 'start': (0, 0),
    'n_holes': 8
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
    frame_buffer = FrameBuffer(device=device, frame_dim=obs_dim)
    frame_buffer_target = FrameBuffer(device=device, frame_dim=obs_dim)

    mean, std = game.get_stats()

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[mean], std=[std]),
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
            state = torch.tensor(game.get_state()).contiguous()  # .permute((2, 0, 1))
            x = preprocess(state).to(device)

            frame_buffer.add_frame(x)
            frame_buffer_target.add_frame(x)

            dqn.train(False)
            action_net = dqn(frame_buffer.get_buffer())

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

            state = torch.tensor(game.get_state()).contiguous()

            x_after = preprocess(state).to(device)
            frame_buffer_target.add_frame(x_after)

            replay_memory.add_sample(frame_buffer.get_buffer(), action.argmax(dim=1), frame_buffer_target.get_buffer(),
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

            # if (s + 1) % 200 == 0:
            #     print(
            #         "Loss at s{}-e{}/{}: {}; current e_rate: {}".format(s + 1, e + 1, n_episodes, loss, current_e_rate))
            #     # frame_buffer.view_buffer()
            #     # frame_buffer_target.view_buffer()
            #     # programPause = raw_input("Press the <ENTER> key to continue...")

            tqdm_.set_description(
                "Loss at s{}-e{}/{}: {}; current e_rate: {}".format(s + 1, e + 1, n_episodes, float(loss),
                                                                    current_e_rate))

            s += 1
            if game.is_terminal:
                print("Terminal game!")
                print("Step before ending:", game.step_count)
                game = GridGame(**game_params)
                frame_buffer = FrameBuffer(frame_dim=obs_dim, device=device)
                frame_buffer_target = FrameBuffer(frame_dim=obs_dim, device=device)
                epoch_reward.append(game.total_reward)
                # break

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

    plt.plot(losses)
    plt.ylabel('loss')
    plt.savefig('loss_per_epoch_{}.pdf'.format(n_episodes))
    plt.show()

    plt.plot(rewards)
    plt.ylabel('rewards')
    plt.savefig('rewards_per_epoch_{}.pdf'.format(n_episodes))
    plt.show()

    if dqn.name == 'target':
        dqn = dqn_target

    # save model for testing
    torch.save(dqn.state_dict(), 'dqn_e{}_game_dim{}.ptd'.format(n_episodes, game_dim))

    test(device=device, dqn=dqn, game_params=game_params, obs_dim=obs_dim, preprocess=preprocess, draw_gif=True)


if __name__ == '__main__':
    main()
