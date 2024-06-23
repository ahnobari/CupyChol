#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse_v2.h>
#include <iostream>

namespace py = pybind11;

void solve_cholesky(int n, int nnz, int* d_csrRowPtrA, int* d_csrColIndA, double* d_csrValA, double* d_b, double* d_x) {
    cusolverSpHandle_t cusolverH = nullptr;
    cusparseMatDescr_t descrA = nullptr;
    int reorder = 0;
    int singularity = 0;
    const double tol = 1e-14;

    cusolverSpCreate(&cusolverH);
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    cusolverSpDcsrlsvchol(cusolverH, n, nnz, descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA, d_b, tol, reorder, d_x, &singularity);

    if (singularity >= 0) {
        std::cerr << "WARNING: A is singular at row " << singularity << std::endl;
    }

    cusparseDestroyMatDescr(descrA);
    cusolverSpDestroy(cusolverH);
}

void solve_cupy_csr(py::object csrRowPtrA, py::object csrColIndA, py::object csrValA, py::object b, py::object x) {
    int* d_csrRowPtrA = reinterpret_cast<int*>(csrRowPtrA.attr("data").attr("ptr").cast<ssize_t>());
    int* d_csrColIndA = reinterpret_cast<int*>(csrColIndA.attr("data").attr("ptr").cast<ssize_t>());
    double* d_csrValA = reinterpret_cast<double*>(csrValA.attr("data").attr("ptr").cast<ssize_t>());
    double* d_b = reinterpret_cast<double*>(b.attr("data").attr("ptr").cast<ssize_t>());
    double* d_x = reinterpret_cast<double*>(x.attr("data").attr("ptr").cast<ssize_t>());

    int n = csrRowPtrA.attr("size").cast<int>() - 1;
    int nnz = csrColIndA.attr("size").cast<int>();

    solve_cholesky(n, nnz, d_csrRowPtrA, d_csrColIndA, d_csrValA, d_b, d_x);
}

PYBIND11_MODULE(cupy_chol, m) {
    m.def("solve_cupy_csr", &solve_cupy_csr, "Solve a linear system using Cholesky decomposition with CuPy arrays");
}
