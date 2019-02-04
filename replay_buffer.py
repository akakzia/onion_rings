import collections
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy

class ReplayBuffer():

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = collections.deque(maxlen=self.buffer_size)

    def push(self, state, action, reward, new_state, overwrite=True):

        """ If maximum buffer size reached , remove old experience """
        if len(self.buffer) == self.buffer_size:
            self.buffer.popleft()

        state = copy.deepcopy(state)
        new_state = copy.deepcopy(new_state)

        """ Add new experience """
        self.buffer.append((state, action, reward, new_state))

    def __len__(self):
        return len(self.buffer)

    def uniform_sample(self, batch_size=1):

        if batch_size > len(self.buffer):
            return np.array([])

        indices = np.random.randint(0, len(self.buffer), batch_size)
        return np.array([self.buffer[i] for i in indices])

    def clear(self):
        self.buffer.clear()

    def visualize(self):

        pos_init = [exp[0][0] for exp in self.buffer]
        pos_final = [exp[3][0] for exp in self.buffer]

        red = 1.0
        green = 0
        blue = 0

        step = 1.0 / (len(self.buffer) / 2)

        plt.set_cmap('RdYlGn')

        for i in range(len(pos_init)):
            x_pos = pos_init[i][0]
            y_pos = pos_init[i][1]
            x_vel = pos_final[i][0] - x_pos
            y_vel = pos_final[i][1] - y_pos

            color = matplotlib.colors.to_hex((red, green, blue))
            plt.plot(x_pos, y_pos, '.', color=color)
            plt.arrow(x_pos, y_pos, x_vel, y_vel, color="blue", width=0.001, head_width=0.008)

            if green < 1.0:
                green = min(green + step, 1.0)
            else:
                red = max(red - step, 0)


        plt.show()

