import cupy as cp
import numpy as np
from cupychol import cuchol_solve
from cupychol.wrapper import compute_ordering
from scipy.io import mmread
from scipy.sparse import csr_matrix
import time

# Example usage
if __name__ == '__main__':
    import scipy.sparse as sp

    data = np.load("nd6k.npz")
    A = csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])
    # Find Ordering while still on CPU to avoid copying the matrix to GPU (This will happen by deafult uf reorder=False is not set)
    order, inv_order = compute_ordering(A)
    print("Converting to cupy CSR matrix")
    A_cupy = cp.sparse.csr_matrix(A)
    A_cupy = A_cupy[order][:, order]
    print("Done")
    # Right-hand side vector b
    b = cp.ones(A.shape[0], dtype=cp.float64)
    b = b[order]
    input("Press Enter to continue...")
    # Solve the system
    print("Solving the system")
    start = time.time()
    x = cuchol_solve(A_cupy,b, reorder=False)
    end = time.time()
    residual = cp.linalg.norm(A_cupy @ x - b)
    print("Residual:", residual)
    print("Time:", end - start)

    print('Solving using cupy cg with higest precision')
    start = time.time()
    from cupyx.scipy.sparse.linalg import cg as cg_gpu
    x,f = cg_gpu(A_cupy, b, tol=1e-16, atol=1e-16, maxiter = 1e5)
    end = time.time()
    print('f:', f)
    residual = cp.linalg.norm(A_cupy @ x - b)
    print("Residual:", residual)
    print("Time:", end - start)

    #pause the program
    input("Press Enter to continue...")
