import numpy as np
from enum import Enum


class Stability(Enum):
    ESTAVEL = 0, "Estável"
    INSTAVEL = 1, "Instável"
    INSTAVEL_SELA = 2, "Instável sela"
    ASSINTOTICAMENTE_ESTAVEL = 3, "Assintoticamente estável"
    MARGINALMENTE_ESTAVEL = 4, "Marginalmente estável"
    NEUTRO_ESTAVEL = 5, "Neutro estável"
    INDETERMINADA = 6, "Indeterminada"

    def text(self):
        return self.value[1]

    def index(self):
        return self.value[0]


class PredatorPrey:

    def __init__(self, b, p, r, d):
        self.b = b  # a
        self.p = p  # alpha
        self.r = r  # gamma
        self.d = d  # c

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
        return np.array([[0, 0], [self.d / self.r, self.b / self.p]])

    def jacobian(self, x, y):
        return np.array(
            [[self.b - self.p * y, -self.p * x], [self.r * y, -self.d + self.r * x]]
        )

    def stability(self, x, y):
        if isinstance(x, np.ndarray):
            stabilities = []
            for i in zip(x, y):
                stabilities.append((i, self.stability(i[0], i[1])))
            return stabilities

        J = self.jacobian(x, y)
        auto_valores, auto_vetores = np.linalg.eig(J)
        parte_real = auto_valores.real
        parte_imaginaria = auto_valores.imag

        stability = self.determine_stability(parte_real, parte_imaginaria)
        return auto_valores, auto_vetores, stability

    def determine_stability(self, parte_real, parte_imaginaria):
        if np.all(parte_real < 0):
            return Stability.ASSINTOTICAMENTE_ESTAVEL
        elif np.all(parte_real == 0):
            if np.any(parte_imaginaria != 0):
                return Stability.MARGINALMENTE_ESTAVEL
            else:
                return Stability.NEUTRO_ESTAVEL
        elif np.all(parte_real <= 0) and np.any(parte_real == 0):
            return Stability.ESTAVEL
        elif np.all(parte_real > 0):
            return Stability.INSTAVEL
        elif np.any(parte_real > 0) and np.any(parte_real < 0):
            return Stability.INSTAVEL_SELA
        else:
            return Stability.INDETERMINADA
