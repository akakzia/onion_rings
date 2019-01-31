import collections
import numpy as np

class ReplayBuffer():

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = collections.deque(maxlen=self.buffer_size)

    def push(self, state, action, reward, new_state, overwrite=True):

        """ If maximum buffer size reached , remove old experience """
        if len(self.buffer) == self.buffer_size:
            self.buffer.popleft()

        """ Add new experience """
        self.buffer.append((state, action, reward, new_state))

    def __len__(self):
        return len(self.buffer)

    def uniform_sample(self, batch_size=1):
        indices = np.random.randint(0, len(self.buffer), batch_size)
        return np.array([self.buffer[i] for i in indices])

    def clear(self):
        self.buffer.clear()
