import numpy as np

class GBM:
    '''Geometric Brownian Motion under risk-neutral world'''
    def __init__(self, mu: float, sigma: float):
        self.mu = mu #drift
        self.sigma = sigma #volatility

    def step(self, s: np.ndarray, dt: float, z: np.ndarray) -> np.ndarray:
        '''Compute one time-step forward'''
        return s * np.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z)
    
