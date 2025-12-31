import numpy as np
def compute_force(phi, T=1.0):
    # F = T * Grad(S)
    gy, gx = np.gradient(phi)
    return T * gx, T * gy