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
import argparse


def cat(*args):
    t_list = [*args]
    if t_list[1] != None:
        return torch.cat(t_list, dim=0)
    else:
        return t_list[0]


game_dim = 16
gamma = 0.85
e_rate_start = 0.90
e_rate_end = 0.1
swap_freq = 2
exploration_stop = 0.25
batch_size = 32
save_plots = True
save_checkpoint = False
ap = argparse.ArgumentParser()

# ap.add_argument("-dv", "--device", type=str, default='cpu',
#                help="Select between 'gpu' or 'cpu'. If cuda is not available, it will run on CPU by default.")
ap.add_argument("-d", "--doubleq", action='store_true', default=False,
                help="Use double Q-learning")
ap.add_argument("-g", "--gpu", action='store_true',
                help="Use GPU acceleration.")

args = vars(ap.parse_args())

use_dql = args['doubleq']
gpu_acc = args['gpu']
if gpu_acc:
    device = 'cuda'
else:
    device = 'cpu'

use_batch_norm = True

n_episodes = 500

game_params = {
    'dim': game_dim,
    'n_holes': game_dim
}

description = "gdim-{}_gamma-{}_nepisodes-{}_explorationstop-{}_b-{}_dql-{}".format(game_dim, gamma, n_episodes,
                                                                                    exploration_stop, batch_size,
                                                                                    use_dql)


def main():
    dqn = MlpDQN(input_dim=game_dim ** 2 * 3, use_batch_norm=use_batch_norm)
    if use_dql:
        dqn_target = MlpDQN(input_dim=game_dim ** 2 * 3, use_batch_norm=use_batch_norm)
    else:
        dqn_target = dqn

    game = GridGame(**game_params)

    dqn.__setattr__('name', 'net')
    dqn_target.__setattr__('name', 'target')

    preprocess = torchvision.transforms.Compose([
        torch.Tensor
    ])

    opt = torch.optim.Adam(lr=1e-4, params=dqn.parameters(), weight_decay=1e-6)
    if use_dql:
        target_opt = torch.optim.Adam(lr=1e-4, params=dqn_target.parameters(), weight_decay=1e-6)

    dqn.to(device)
    dqn_target.to(device)

    replay_memory = ReplayMemory(device)

    lambda1 = lambda e: max(
        e_rate_start * (1 - e / n_episodes * 1 / exploration_stop) + e_rate_end * e / n_episodes * 1 / exploration_stop,
        e_rate_end)

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
            state = game.get_state(upscale=False)  # .permute((2, 0, 1))
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

            state = game.get_state(upscale=False)
            x_after = preprocess(state).to(device).unsqueeze(0)

            replay_memory.add_sample(x, action.argmax(dim=1), x_after,
                                     reward)

            # sample from replay memory
            x_batch, actions_batch, x_then_batch, reward_batch = replay_memory.get_sample(batch_size)

            if len(x_batch) > 1:
                dqn.train(True)
            else:
                dqn.train(False)
            Q_predicted = dqn(x_batch)

            with torch.no_grad():
                dqn_target.train(False)
                dqn.train(False)
                argmax_actions = dqn(x_then_batch).argmax(dim=1)
                Q_then_predicted = dqn_target(x_then_batch).gather(1, argmax_actions.unsqueeze(-1))

            gt_non_terminal = reward_batch + gamma * Q_then_predicted  # .max(dim=1)[0]
            gt_terminal = reward_batch
            gt = torch.where(reward_batch != 2, gt_non_terminal, gt_terminal)

            loss = (gt - torch.gather(Q_predicted, 1, actions_batch.unsqueeze(-1))) ** 2
            loss = loss.mean()
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

        if ((e + 1) % 100 == 0) and save_checkpoint:
            torch.save({
                'model': dqn.state_dict(),
                'opt': opt.state_dict(),
                'epoch': e,
            }, 'dqn_training_checkpoint_e{}_{}.ptd'.format(e, description))


        if ((e + 1) % swap_freq == 0) and use_dql:
            print("SWAPPING NETWORKS & OPTIMIZERS!")
            dqn, dqn_target = dqn_target, dqn
            opt, target_opt = target_opt, opt

    print("Training time: {} minutes".format(-(train_t0 - time.time()) // 60))

    plt.plot(losses[10:])
    plt.ylabel('loss')
    if save_plots:
        plt.savefig('loss_per_epoch_{}.pdf'.format(description))

    plt.close()

    plt.plot(rewards)
    plt.ylabel('rewards')
    if save_plots:
        plt.savefig('rewards_per_epoch_{}.pdf'.format(description))

    if dqn.name == 'target':
        dqn = dqn_target

    # save model for testing
    filename = 'dqn_{}.ptd'.format(description)
    print("Saving model to: ", filename)
    torch.save(dqn.state_dict(), filename)

    test(device=device, dqn=dqn, game_params=game_params, preprocess=preprocess, draw_gif=True)


if __name__ == '__main__':
    main()
