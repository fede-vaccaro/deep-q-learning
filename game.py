import random

import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt


def manhattan(a, b):
    ax, ay = a
    bx, by = b

    return abs(ax - bx) + abs(ay - by)


class GridGame:
    def __init__(self, dim=16, start=None, finish=None):
        state = np.ones((dim, dim, 3), dtype='float32')

        self.side_dim = dim

        if (not start) and (not finish):
            start, finish = self.init_randomized_start()
        elif not start:
            start = (0, 0)
        elif not finish:
            finish = (dim - 1, dim - 1)

        state[start] = [1.0, 0.0, 0.0]
        state[finish] = [0.0, 1.0, 0.0]

        self.state = state
        self.start = start
        self.current = start
        self.finish = finish
        self.total_reward = 0

        self.max_steps = abs(start[0] - finish[0]) + abs(start[1] - finish[1])
        self.step_count = 0
        self.is_terminal = False
        self.visited_cells = set()

    def init_randomized_start(self):
        '''
        initialize start and finish positions on the perimeter, at high distance
        :return:
        '''
        # perimeter:
        # 0, dim - 1 -> upper side -> side 0
        # dim, 2*dim-1 -> right side -> side 1
        # 2*dim, 3*dim-1 -> lower side -> side 2
        # 3*dim, 4*dim-1 -> left side -> side 3

        start_perimeter_pos = random.randint(0, self.side_dim * 4 - 1)
        start_side = start_perimeter_pos // self.side_dim
        start_pos_on_side = start_perimeter_pos % self.side_dim

        start = self.get_init_position(start_pos_on_side, start_side)

        finish_perimeter_pos = (start_perimeter_pos + random.randint(self.side_dim, self.side_dim * 2) - 1) % (
                self.side_dim * 4 - 1)
        finish_side = finish_perimeter_pos // self.side_dim
        finish_pos_on_side = finish_perimeter_pos % self.side_dim

        finish = self.get_init_position(finish_pos_on_side, finish_side)

        return start, finish

    def get_init_position(self, pos_on_side, side):
        row = -1
        col = -1

        if side == 0:
            row = 0
        elif side == 1:
            col = self.side_dim - 1
        elif side == 2:
            row = self.side_dim - 1
        elif side == 3:
            col = 0

        if row == -1:
            row = pos_on_side
        if col == -1:
            col = pos_on_side
        start = (row, col)
        return start

    def visualize_state(self):
        # img = Image.fromarray((self.state * 255.0).astype('uint8'), 'RGB')
        # img.show()
        plt.imshow(self.state)
        plt.show()

    def action(self, a):
        self.state[self.current] = (1.0, 1.0, 1.0)

        action_index = a
        dim = self.side_dim

        old_state = self.current
        self.visited_cells.add(old_state)

        if action_index == 0:  # up
            self.current = max(self.current[0] - 1, 0), self.current[1]
        elif action_index == 1:  # right
            self.current = self.current[0], min(self.current[1] + 1, dim - 1)
        elif action_index == 2:  # down
            self.current = min(self.current[0] + 1, dim - 1), self.current[1]
        elif action_index == 3:  # left
            self.current = self.current[0], max(self.current[1] - 1, 0)

        if self.current == self.finish:
            reward = manhattan(self.start, self.finish)
            self.state[self.current] = (0.0, 0.0, 1.0)
            self.is_terminal = True
        elif self.current == old_state:  # or self.current in self.visited_cells:
            self.state[self.current] = (1.0, 0.0, 0.0)
            # reward = -manhattan(self.current, self.finish)/self.dim*2.0
            reward = -2
        else:
            self.state[self.current] = (1.0, 0.0, 0.0)
            reward = manhattan(old_state, self.finish) - manhattan(self.current, self.finish)
            # reward = -1

        self.step_count += 1
        self.total_reward += reward
        return reward

    def get_stats(self):
        state_grey = self.get_state()
        return state_grey.flatten().mean(), state_grey.flatten().std()

    def get_state(self, rgb=False):
        if rgb:
            return self.state
        r, g, b = self.state[:, :, 0], self.state[:, :, 1], self.state[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray[np.newaxis, :]


if __name__ == '__main__':
    game = GridGame()

    # action_1 = np.array(3)
    # action_2 = np.array(2)
    # action_3 = np.array(2)
    # action_4 = np.array(2)
    #
    # game.action(action_1)
    # game.action(action_2)
    # game.action(action_3)
    # game.action(action_4)

    game.visualize_state()
