import numpy as np
from mcopt.models.gbm import GBM

import numpy as np
from mcopt.models.gbm import GBM


def simulate_one_path(model: GBM,
                      s0: float,
                      T: float,
                      steps: int,
                      rng: np.random.Generator) -> np.ndarray:
    """
    Simulate a SINGLE GBM price path.

    Returns an array of length steps+1:
    [S0, S_t1, S_t2, ..., S_T]
    """
    dt = T / steps           # size of each time step
    prices = np.empty(steps + 1, dtype=float)

    prices[0] = s0           # starting value

    s = s0                   # current price
    for t in range(steps):
        z = rng.standard_normal()   # one random shock
        s = model.step(np.array([s]), dt, np.array([z]))[0]
        prices[t + 1] = s

    return prices



def simulate_paths(model: GBM,
                   s0: float,
                   T: float,
                   steps: int,
                   n_paths: int,
                   rng: np.random.Generator) -> np.ndarray:
    """
    Simulate MANY GBM price paths.

    Returns an array with shape (n_paths, steps+1):
    each row = one path.
    """
    paths = np.empty((n_paths, steps + 1), dtype=float)

    for i in range(n_paths):
        paths[i] = simulate_one_path(model, s0, T, steps, rng)

    return paths
