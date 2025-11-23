import numpy as np

from mcopt.models.gbm import GBM
from mcopt.simulation.paths import simulate_paths

S0 = 100.0      # initial stock price
K = 100.0       # strike price
r = 0.05        # risk-free rate
sigma = 0.2     # volatility
T = 1.0         # maturity in years
steps = 252     # time steps
n_paths = 50  # number of Monte Carlo paths

rng = np.random.default_rng(42)
model = GBM(mu=r, sigma=sigma)

paths = simulate_paths(model, S0, T, steps, n_paths, rng)
# extract final price
ST = paths[:, -1]
# compute payoff
payoffs = np.maximum(ST - K, 0.0)
# discount average payoff back to today
disc = np.exp(-r * T)
price_estimate = disc * np.mean(payoffs)
# estimate MC standard error
std_payoff = np.std(payoffs, ddof=1)
stderr = disc * std_payoff / np.sqrt(n_paths)

print(f"Estimated call price: {price_estimate:.4f}")
print(f"Standard error:       {stderr:.4f}")
