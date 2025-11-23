# examples/one_path.py

import numpy as np
import matplotlib.pyplot as plt

from mcopt.models.gbm import GBM
from mcopt.simulation.paths import simulate_one_path

# --- parameters ---
S0 = 100.0
r = 0.05
sigma = 0.2
T = 1.0          # 1 year
steps = 252      # 252 trading days

rng = np.random.default_rng(42)
model = GBM(mu=r, sigma=sigma)

# --- simulate one path ---
path = simulate_one_path(model, S0, T, steps, rng)

print("Simulated path values:")
print(path)

# --- build time axis ---
times = np.linspace(0.0, T, steps + 1)  # 0, 1/252, ..., 1.0

# --- plot ---
plt.plot(times, path)
plt.xlabel("Time (years)")
plt.ylabel("Stock Price")
plt.title("One GBM Price Path")
plt.grid(True)
plt.show()
