import argparse

import numpy as np
import torchvision
from tqdm import tqdm

from game import GridGame
from model import *
from PIL import Image


def test(device, dqn, preprocess, game_params, draw_gif=True):
    # play a game and show how the agent acts!


    game = GridGame(**game_params)
    states = []
    max_steps = 1000
    dqn.train(False)
    with torch.no_grad():
        for i in range(max_steps):
            if draw_gif:
                state_rgb = game.get_state(upscale=True).resize((400, 400), Image.NEAREST)
                states.append(state_rgb)

            if game.is_terminal:
                if draw_gif:
                    print("Agent won in {} steps!".format(i))
                break

            state = game.get_state(upscale=False)

            x = preprocess(state).to(device).unsqueeze(0)
            if random.uniform(0.0, 1.0) < 0.1:
                action = random.randint(0, 3)
            else:
                action = dqn(x).argmax()

            game.action(action)
    if draw_gif:
        states[0].save('match_dim{}.gif'.format(game_params['dim']),
                       save_all=True, append_images=states[1:], optimize=False, duration=150, loop=0)
        print("Total reward:", game.total_reward)
    return game.total_reward


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument("-f", "--filename", type=str,
                    help="Specify model filename.")
    ap.add_argument("-g", "--gpu", action='store_true',
                    help="Use GPU acceleration.")

    args = vars(ap.parse_args())
    filename = args['filename']
    gpu_acc = args['gpu']
    if gpu_acc:
        device = 'cuda'
    else:
        device = 'cpu'

    game_dim = 16
    print("Testing", filename)

    game_params = {
        'dim': game_dim,
        # 'start': (0, 0),
        'n_holes': 16
    }

    dqn = MlpDQN(input_dim=game_dim ** 2 * 3, use_batch_norm=True)
    weights = torch.load(filename)
    dqn.load_state_dict(weights)
    dqn.to(device)

    preprocess = torchvision.transforms.Compose([
        torch.Tensor
    ])
    draw_gif = False

    if not draw_gif:
        rewards = []
        for i in tqdm(range(1000)):
            r = test(device=device, dqn=dqn, game_params=game_params, preprocess=preprocess,
                     draw_gif=draw_gif)
            rewards += [r]

        rewards = np.array(rewards)
        print("Mean reward: {}".format(rewards.mean()))
        print("Num positive reward: {}/{}".format(len(rewards[rewards > 0]), len(rewards)))
        print("Num finished matches: {}/{}".format(len(rewards[rewards >= -500.0]), len(rewards)))
    else:
        print("Total reward", test(device=device, dqn=dqn, game_params=game_params, preprocess=preprocess,
                                   draw_gif=draw_gif))
