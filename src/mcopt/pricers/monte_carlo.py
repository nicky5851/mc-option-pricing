import numpy as np
from mcopt.simulation.paths import simulate_paths

def price(model,
          product,
          S0: float,
          T: float,
          steps: int,
          n_paths: int,
          r: float,
          rng: np.random.Generator) -> dict:
    """Generic Monte Carlo pricer for path-dependent products.

    model: object with a .step(...) method (e.g. GBM)
    product: object with a .payoff(paths) method (e.g. EuropeanCall)
    """
    paths = simulate_paths(model, S0, T, steps, n_paths, rng)
    payoffs = product.payoff(paths)
    disc = np.exp(-r * T)
    est = disc * float(np.mean(payoffs))
    # standard error
    std = float(np.std(payoffs, ddof=1))
    stderr = disc * std / np.sqrt(n_paths)

    return {"price": est, "stderr": stderr}

