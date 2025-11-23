import numpy as np

from mcopt.models.gbm import GBM
from mcopt.products import EuropeanCall
from mcopt.pricers import price

S0 = 100.0
K = 100.0
r = 0.05
sigma = 0.2
T = 1.0
steps = 252
n_paths = 50_000

rng = np.random.default_rng(42)
model = GBM(mu=r, sigma=sigma)
call = EuropeanCall(K)

result = price(model, call, S0, T, steps, n_paths, r, rng)

print("Result:", result)
print(f"Estimated price: {result['price']:.4f}")
print(f"Std error:       {result['stderr']:.4f}")