import cupy as cp
import numpy as np
import cupy_chol
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

def compute_ordering(A):
    ordr = reverse_cuthill_mckee(A, symmetric_mode=True)
    inv_order = np.argsort(ordr)
    return ordr, inv_order

def solve_linear_system(A, b, reorder=True):
    """
    Solve the linear system Ax = b using Cholesky decomposition with CuPy arrays.
    
    Parameters:
    A (cp.sparse.csr_matrix): The coefficient matrix in CSR format.
    b (cp.ndarray): The right-hand side vector.
    reorder (bool): Whether to use the reverse Cuthill-McKee ordering. This will move the matrix to CPU memory for ordering. Pre ordering can improve the performance of the solver if the sparsity pattern is constant and one reordering is computed.
    
    Returns:
    cp.ndarray: The solution vector x.
    """
    # Validate input types
    if not cp.sparse.isspmatrix_csr(A):
        raise ValueError("Matrix A must be in CSR format")
    
    if not isinstance(b, cp.ndarray):
        raise ValueError("Vector b must be a CuPy ndarray")
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square")
    
    if A.shape[0] != b.size:
        raise ValueError("The size of b must be equal to the number of rows in A")
    
    # Create an empty solution vector x
    x = cp.zeros(b.size, dtype=cp.float64)
    
    if reorder:
        # Compute the reverse Cuthill-McKee ordering
        ordr, inv_order = compute_ordering(A.get())
        
        # Reorder the matrix and right-hand side vector
        A = A[ordr][:, ordr]
        b = b[ordr]
    
    # Call the underlying C++ function
    cupy_chol.solve_cupy_csr(A.indptr, A.indices, A.data, b, x)
    
    if reorder:
        # Reverse the ordering of the solution vector
        x = x[inv_order]
    
    return x