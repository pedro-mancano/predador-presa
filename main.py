import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from RungeKutta import RK4
from PredatorPrey import PredatorPrey

import os

os.makedirs("build", exist_ok=True)


# Definição da classe predador-presa
pp = PredatorPrey(1.5, 0.5, 1, 0.5)
# Encontrado os pontos de equilibrio
equilibrio = pp.equilibrum()
ex = equilibrio[:, 0]
ey = equilibrio[:, 1]

# Encontrado a estabilidade para os pontos
estabilidade = pp.stability(ex, ey)

# Condições iniciais
xi = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
yi = [3.5, 4.3125, 5.125, 5.9375, 6.75, 7.5625, 8.375, 9.1875, 10]
initial_conditions = np.array([xi, yi])

# Runge Kutta de 4 ordem
STEP = 0.01
solver = RK4(initial_conditions, pp.step, STEP)

# Resolvendo até 50s
t, state_output = solver.solve_until(50)
x = state_output[0, :]
y = state_output[1, :]


# Plotando diagrama de fase
fig, ax = plt.subplots()
x_range = np.linspace(0, float(np.max(x)) * 1.1, 20)
y_range = np.linspace(0, float(np.max(y)) * 1.1, 20)
X, Y = np.meshgrid(x_range, y_range)
DX, DY = pp.rate_at(X, Y)

norm = Normalize(vmin=0, vmax=len(x) + 1)
cmap = cm.get_cmap("rainbow")

ax.streamplot(X, Y, DX, DY, density=2, arrowsize=0.5, linewidth=0.5)

ax.axvline(ex[0], color="black", alpha=0.5, linestyle="--", zorder=-1)
ax.axhline(ey[0], color="black", alpha=0.5, linestyle="--", zorder=-1)
ax.axvline(ex[1], color="black", alpha=0.5, linestyle="--", zorder=-1)
ax.axhline(ey[1], color="black", alpha=0.5, linestyle="--", zorder=-1)
ax.scatter(
    ex[0],
    ey[0],
    label=f"Equilibrio\n({ex[0]}, {ey[0]})",
    color=cmap(norm(0)),
    marker="x",
)
ax.scatter(
    ex[1],
    ey[1],
    label=f"Equilibrio\n({ex[1]}, {ey[1]})",
    color=cmap(norm(0)),
    marker="x",
)

for i in range(len(x)):
    ax.plot(
        x[i],
        y[i],
        label=f"({round(initial_conditions[0][i], 2)}, {round(initial_conditions[1][i], 2)})",
        color=cmap(norm(i + 1)),
    )


for ponto in estabilidade:
    eigenvalues, eigenvectors, _ = ponto[1]
    origin = ponto[0]


ax.set_xlabel("Presas")
ax.set_ylabel("Predadores")
ax.set_xlim(0, np.max(x) * 1.1)
ax.set_ylim(0, np.max(y) * 1.1)
ax.legend()

fig.tight_layout(pad=0.2)
fig.savefig("build/presasxpredador.png", dpi=300)

# Presas e Predadores no tempo
fig, ax = plt.subplots(figsize=(8, 3))

ax.plot(
    t,
    x[-1],
    label=f"Presas",
    color="#ff6961",
)

ax.scatter([0], [x[-1][0]], label=f"{(x[-1][0])}", marker="^", color="#ff6961")

ax.plot(
    t,
    y[-1],
    label=f"Predadores",
    color="#A7C7E7",
)

ax.scatter([0], [y[-1][0]], label=f"{(y[-1][0])}", marker="o", color="#A7C7E7")

ax.set_xlabel("Tempo")
ax.set_ylabel("Presa, Predador")
ax.legend()

fig.tight_layout(pad=0.2)
fig.savefig("build/tempo.png", dpi=300)


# Diagrama de estabilidade
fig, ax = plt.subplots(figsize=(6, 4))

ax.axvline(0, color="black", alpha=0.5, linestyle="--", zorder=-1)
ax.axhline(0, color="black", alpha=0.5, linestyle="--", zorder=-1)

for ponto in estabilidade:
    x = ponto[1][0].real
    y = ponto[1][0].imag
    ax.scatter(x, y, label=f"({ponto[0][0]}, {ponto[0][1]}) - {ponto[1][2].text()}")

ax.set_xlabel("Parte Real")
ax.set_ylabel("Parte Imaginária")
ax.legend()

fig.tight_layout(pad=0.2)
fig.savefig("build/estabilidade.png", dpi=300)
