import numpy as np
def get_entropy(phi):
    p = np.abs(phi)**2 / np.sum(np.abs(phi)**2)
    return -np.sum(p * np.log(p + 1e-15))