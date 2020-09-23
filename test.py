import torchvision

from game import GridGame
from model import *
from PIL import Image


def test(device, dqn, preprocess, obs_dim, game_params, draw_gif=True):
    # play a game and show how the agent acts!
    game = GridGame(**game_params)
    frame_buffer = FrameBuffer(device=device, frame_dim=obs_dim)
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
            if random.uniform(0.0, 1.0) < 0.1 / 2.0:
                action = random.randint(0, 3)
            else:
                action = dqn(frame_buffer.get_buffer()).argmax()

            game.action(action)
    if draw_gif:
        states[0].save('match_dim{}.gif'.format(game_params['dim']),
                       save_all=True, append_images=states[1:], optimize=False, duration=150, loop=0)
    print("Total reward:", game.total_reward)


if __name__ == '__main__':
    device = 'cuda'
    obs_dim = 84
    game_dim = 8
    model_name = 'dqn_e500_game_dim8.ptd'

    game_params = {
        'dim': game_dim,
        # 'start': (0, 0),
        'n_holes': 0
    }

    dqn = DQN(input_dim=obs_dim, use_batch_norm=False)
    weights = torch.load(model_name)
    dqn.load_state_dict(weights)
    dqn.to(device)

    game = GridGame(dim=game_dim)
    frame_buffer = FrameBuffer(device=device, frame_dim=obs_dim)

    mean, std = game.get_stats()

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[mean], std=[std]),
    ])

    test(device=device, dqn=dqn, game_params=game_params, obs_dim=obs_dim, preprocess=preprocess, draw_gif=True)
