import numpy as np
def simulate_yp(phi):
    energy = 0.5 * phi**2
    n_he = np.sum(energy > 0.8)
    n_h = np.sum(energy <= 0.8)
    return (4 * n_he) / (4 * n_he + n_h)