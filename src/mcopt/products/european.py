import numpy as np

class EuropeanCall:
    """European call option: payoff = max(S_T - K, 0)."""
    def __init__(self, strike: float):
        self.K = float(strike)
    def payoff (self, paths: np.ndarray) -> np.ndarray:
        '''Compute payoff from simulated paths.
        paths: array of shape (n_paths, steps+1)
        returns: array of shape (n_paths)
        '''
        ST = paths[:, -1]
        return np.maximum(ST - self.K, 0.0)
    


    

