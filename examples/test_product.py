import numpy as np

from mcopt.models.gbm import GBM
from mcopt.simulation.paths import simulate_paths
from mcopt.products.european import EuropeanCall

S0 = 100.0
K = 100.0
r = 0.05
sigma = 0.2
T = 1.0
steps = 252
n_paths = 5_000

rng = np.random.default_rng(42)
model = GBM(mu=r, sigma=sigma)

paths = simulate_paths(model, S0, T, steps, n_paths, rng)

call = EuropeanCall(K)
payoffs = call.payoff(paths)

print("Payoffs shape:", payoffs.shape)
print("First 5 payoffs:", payoffs[:5])