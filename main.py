import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from RungeKutta import RK4
from PredatorPrey import PredatorPrey
import os

os.makedirs("build", exist_ok=True)

pp = PredatorPrey(1.5, 0.5, 1, 0.5)
ex, ey = pp.equilibrum()

xi = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
yi = np.linspace(3.5, 10, 9)

initial_conditions = np.array([xi, yi])


STEP = 0.01
solver = RK4(initial_conditions, pp.step, STEP)
t, state_output = solver.solve_until(50)
x = state_output[0, :]
y = state_output[1, :]


norm = Normalize(vmin=0, vmax=len(x) + 1)
cmap = cm.get_cmap("rainbow")

fig, ax = plt.subplots()

x_range = np.linspace(0, float(np.max(x)) * 1.1, 20)
y_range = np.linspace(0, float(np.max(y)) * 1.1, 20)
X, Y = np.meshgrid(x_range, y_range)
DX, DY = pp.rate_at(X, Y)


plt.streamplot(X, Y, DX, DY, density=2, arrowsize=0.5, linewidth=0.5)
plt.axvline(ex, color="black", alpha=0.5, linestyle="--", zorder=-1)
plt.axhline(ey, color="black", alpha=0.5, linestyle="--", zorder=-1)
plt.scatter(ex, ey, label=f"Equilibrio\n({ex}, {ey})", color=cmap(norm(0)), marker="x")
for i in range(len(x)):
    plt.plot(
        x[i],
        y[i],
        label=f"({round(initial_conditions[0][i], 2)}, {round(initial_conditions[1][i], 2)})",
        color=cmap(norm(i + 1)),
    )


ax.set_xlabel("Presas")
ax.set_ylabel("Predadores")
ax.set_title("")
ax.set_xlim(0, np.max(x) * 1.1)
ax.set_ylim(0, np.max(y) * 1.1)
ax.legend()
fig.tight_layout(pad=0.2)
fig.savefig("build/presasxpredador.png", dpi=300)

fig, ax = plt.subplots()

plt.plot(
    t,
    x[-1],
    label=f"Presas",
    color="#ff6961",
)

plt.plot(
    t,
    y[-1],
    label=f"Predadores",
    color="#A7C7E7",
)


ax.set_xlabel("Tempo")
ax.set_ylabel("Presa, Predador")
ax.legend()

fig.tight_layout(pad=0.2)
fig.savefig("build/tempo.png", dpi=300)
