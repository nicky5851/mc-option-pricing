import numpy as np
import matplotlib.pyplot as plt

from mcopt.models.gbm import GBM
from mcopt.simulation.paths import simulate_paths

S0 = 100.0
r = 0.05
sigma = 0.2
T = 1.0         # one year
steps = 252     # trading days
n_paths = 50000 # no. of paths

rng = np.random.default_rng(42)
model = GBM(mu=r, sigma=sigma)

paths = simulate_paths(model, S0, T, steps, n_paths, rng)

times = np.linspace(0.0, T, steps + 1)

for i in range(20):
    plt.plot(times, paths[i], alpha=0.7)

plt.xlabel('Time (years)')
plt.ylabel('Stock Price')
plt.title(f'{n_paths} GBM Price Paths')
plt.grid(True)
plt.show()