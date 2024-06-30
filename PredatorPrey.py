import numpy as np


class PredatorPrey:

    def __init__(self, b, p, r, d):
        self.b = b
        self.p = p
        self.r = r
        self.d = d

    def rate_at(self, x, y):
        dx = (self.b - self.p * y) * x
        dy = (self.r * x - self.d) * y
        return dx, dy

    def step(self, time, state):
        x = state[0, :]
        y = state[1, :]

        dx, dy = self.rate_at(x, y)

        state[0, :] = dx
        state[1, :] = dy

        return state

    def equilibrum(self):
        return (self.d / self.r, self.b / self.p)

    def jacob(self, x, y):
        return np.array(
            [
                [(self.b - self.p * y), (0 - self.p * x)],
                [(self.r * y - 0), (self.r * x - self.d)],
            ]
        )
