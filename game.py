import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt


def manhattan(a, b):
    ax, ay = a
    bx, by = b

    return abs(ax - bx) + abs(ay - by)

class GridGame:
    def __init__(self, dim=16, start=(0, 0), finish=None):
        state = np.ones((dim, dim, 3), dtype='float32')
        state[start] = [1.0, 0.0, 0.0]

        if not finish:
            finish = (dim - 1, dim - 1)

        state[finish] = [0.0, 1.0, 0.0]

        self.dim = dim
        self.state = state
        self.start = start
        self.current = start
        self.finish = finish

        self.max_steps = abs(start[0] - finish[0]) + abs(start[1] - finish[1])
        self.step_count = 0
        self.is_terminal = False
        self.visited_cells = set()

    def visualize_state(self):
        #img = Image.fromarray((self.state * 255.0).astype('uint8'), 'RGB')
        #img.show()
        plt.imshow(self.state)
        plt.show()

    def action(self, a):
        self.state[self.current] = (1.0, 1.0, 1.0)

        action_index = a
        dim = self.dim

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
            # reward = manhattan(self.start, self.finish)
            reward = 1
            self.state[self.current] = (0.0, 0.0, 1.0)
            self.is_terminal = True
        elif self.current == old_state or self.current in self.visited_cells:
            self.state[self.current] = (1.0, 0.0, 0.0)
            #reward = -manhattan(self.current, self.finish)/self.dim*2.0
            reward = -1
        else:
            self.state[self.current] = (1.0, 0.0, 0.0)
            # reward = -manhattan(self.current, self.finish)/self.dim*0.5
            reward = -1

        self.step_count += 1

        return reward


if __name__ == '__main__':
    game = GridGame()

    action_1 = np.array([0, 0, 0, 1])
    action_2 = np.array([0, 1, 0, 0])
    action_3 = np.array([0, 1, 0, 0])
    action_4 = np.array([0, 0, 1, 0])

    game.action(action_1)
    game.action(action_2)
    game.action(action_3)
    game.action(action_4)

    game.visualize_state()
