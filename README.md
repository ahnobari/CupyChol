# CupyChol

CupyChol is a Python package for solving linear systems using Cholesky decomposition with CuPy arrays. It leverages CUDA and cuSOLVER to provide efficient solutions for large, sparse matrices on the GPU.

## Features

- Solve linear systems of the form `Ax = b` using Cholesky decomposition.
- Works with CuPy arrays, keeping data on the GPU for maximum efficiency.
- Utilizes CUDA and cuSOLVER for high performance.

## Installation

You can install CupyChol from PyPI:

```bash
pip install CupyChol
