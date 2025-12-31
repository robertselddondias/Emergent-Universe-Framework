import numpy as np
def generate_phi(size=100):
    state = np.random.normal(0, 1, (size, size))
    return state / np.sum(np.abs(state))