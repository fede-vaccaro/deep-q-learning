import numpy as np
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR

from game import GridGame
from model import *
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
n_episodes = 500
gamma = 0.95
e_rate_start = 0.90
e_rate_end = 0.1


def main():
    dqn = DQN(input_dim=game_dim, use_batch_norm=False)
    game = GridGame(dim=game_dim)

    device = 'cuda'
    frame_buffer = FrameBuffer(device=device, frame_dim=game_dim)
    frame_buffer_target = FrameBuffer(device=device, frame_dim=game_dim)

    mean, std = game.get_stats()

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[mean], std=[std]),
    ])

    opt = torch.optim.Adam(lr=1e-4, params=dqn.parameters())

    dqn.to(device)

    replay_memory = ReplayMemory(device)

    exploration_stop = 0.5
    exploration_stop = 1 / exploration_stop
    lambda1 = lambda e: max(e_rate_start * (1 - e / n_episodes * 1 / exploration_stop) + e_rate_end * e / n_episodes * 1 / exploration_stop,
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

            current_e_rate = max(e_rate_start * (1 - e / n_episodes * 1.5) + e_rate_end * e / n_episodes * 1.5,
                                 e_rate_end)
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
                dqn.train(False)
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
                frame_buffer = FrameBuffer(frame_dim=game_dim, device=device)
                frame_buffer_target = FrameBuffer(frame_dim=game_dim, device=device)
                break

        epoch_loss = np.array(epoch_loss).mean()
        losses += [epoch_loss]

        if (e + 1) % 100 == 0:
            torch.save({
                'model': dqn.state_dict(),
                'opt': opt.state_dict(),
                'epoch': e,
            }, 'dqn_training_e{}_game_dim{}.ptd'.format(n_episodes, game_dim))

            # scheduler.step()

    plt.plot(losses)
    plt.ylabel('loss')
    plt.savefig('training_{}.pdf'.format(n_episodes))
    plt.show()

    # save model for testing
    torch.save(dqn.state_dict(), 'dqn_e{}_game_dim{}.ptd'.format(n_episodes, game_dim))

    # play a game and show how the agent acts!
    game = GridGame(dim=game_dim)
    frame_buffer = FrameBuffer(device=device, frame_dim=game_dim)

    states = []
    max_steps = 1000

    dqn.train(False)
    with torch.no_grad():
        for i in range(max_steps):
            state_rgb = Image.fromarray((game.get_state(rgb=True) * 255.0).astype('uint8'), 'RGB').resize((400, 400))
            states.append(state_rgb)

            if game.is_terminal:
                print("Agent won in {} steps!".format(i))
                break

            state = torch.tensor(game.get_state()).contiguous()

            x = preprocess(state).to(device)
            frame_buffer.add_frame(x)
            if random.uniform(0.0, 1.0) < e_rate_end / 2.0:
                action = random.randint(0, 3)
            else:
                action = dqn(frame_buffer.get_buffer()).argmax()

            game.action(action)

    states[0].save('match_{}_dim{}.gif'.format(n_episodes, game_dim),
                   save_all=True, append_images=states[1:], optimize=False, duration=150, loop=0)


if __name__ == '__main__':
    main()
