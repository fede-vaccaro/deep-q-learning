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
                state_rgb = game.get_state().resize((400, 400), Image.NEAREST)
                states.append(state_rgb)

            if game.is_terminal:
                if draw_gif:
                    print("Agent won in {} steps!".format(i))
                break

            state = game.get_state()

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
    device = 'cuda'
    game_dim = 16
    model_name = 'dqn_e500_game_dim16_.ptd'
    print("Testing", model_name)

    game_params = {
        'dim': game_dim,
        # 'start': (0, 0),
        'n_holes': 16
    }

    dqn = DQN(input_dim=game_dim**2*2, use_batch_norm=True)
    weights = torch.load(model_name)
    dqn.load_state_dict(weights)
    dqn.to(device)

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    rewards = []
    for i in tqdm(range(1000)):
        r = test(device=device, dqn=dqn, game_params=game_params, preprocess=preprocess,
                 draw_gif=False)
        rewards += [r]

    rewards = np.array(rewards)
    print("Mean reward: {}".format(rewards.mean()))
    print("Num positive reward: {}/{}".format(len(rewards[rewards > 0]), len(rewards)))
