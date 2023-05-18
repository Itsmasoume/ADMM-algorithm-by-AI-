# ADMM
import numpy as np

from numba import njit, prange

@njit(parallel=True)

def admm_solver(A, b, rho, max_iter):

    n = A.shape[1]  # Number of variables

    m = A.shape[0]  # Number of constraints

    x = np.zeros(n)  # Variable to optimize

    z = np.zeros(n)  # Auxiliary variable

    u = np.zeros(n)  # Dual variable

    for _ in prange(max_iter):

        # Update x

        x = np.linalg.solve(A.T @ A + rho * np.eye(n), A.T @ b + rho * (z - u))

        # Update z

        z_prev = np.copy(z)

        z = np.maximum(x + u, 0)

        # Update u

        u += x - z

    return x

# Example usage

A = np.array([[1, 2], [3, 4], [5, 6]])  # Coefficient matrix

b = np.array([3, 7, 11])  # Constraint values

rho = 1.0  # Penalty parameter

max_iter = 100  # Maximum number of iterations

solution = admm_solver(A, b, rho, max_iter)

print("Optimal solution:", solution)
