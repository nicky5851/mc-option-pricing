import numpy as np
from mcopt.models.gbm import GBM

print('Manual GBM step test')

# 1. parameters
S0 = 100.0      # current price
r = 0.05        # risk-free drift
sigma = 0.20    # volatility
dt = 1.0/252    # one day trading

gbm = GBM(mu=r, sigma=sigma)

# 2. try fixed shock
for z in [0.0, 1.0, -1.0, 0.5, -0.5]:
    S1 = gbm.step(np.array([S0]), dt, np.array([z]))[0]
    print(f'z={z:+.1f} -> S1={S1:.4f}')

#3. random shock from standard normal
rng = np.random.default_rng(42)
Z = rng.standard_normal(5) # 5 random draws ~ N(0,1)
S1s = gbm.step(np.full(5, S0), dt, Z)
print('Random z:', np.round(Z, 3))
print('Random step S1:', np.round(S1s, 4))