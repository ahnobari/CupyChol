from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os
from typing import List, Optional

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the `get_include()` method can be invoked."""

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

def get_path(key: str) -> List[str]:
    return os.environ.get(key, '').split(os.pathsep)

def print_warning(*lines: str) -> None:
    print('**************************************************')
    for line in lines:
        print('*** WARNING: %s' % line)
    print('**************************************************')

def search_on_path(filenames: List[str]) -> Optional[str]:
    for p in get_path('PATH'):
        for filename in filenames:
            full = os.path.join(p, filename)
            if os.path.exists(full):
                return os.path.abspath(full)
    return None

def get_cuda_path():

    nvcc_path = search_on_path(('nvcc', 'nvcc.exe'))
    cuda_path_default = None
    if nvcc_path is not None:
        cuda_path_default = os.path.normpath(
            os.path.join(os.path.dirname(nvcc_path), '..'))

    cuda_path = os.environ.get('CUDA_PATH', '')  # Nvidia default on Windows
    if len(cuda_path) > 0 and cuda_path != cuda_path_default:
        print_warning(
            'nvcc path != CUDA_PATH',
            'nvcc path: %s' % cuda_path_default,
            'CUDA_PATH: %s' % cuda_path)

    if os.path.exists(cuda_path):
        _cuda_path = cuda_path
    elif cuda_path_default is not None:
        _cuda_path = cuda_path_default
    elif os.path.exists('/usr/local/cuda'):
        _cuda_path = '/usr/local/cuda'
    else:
        _cuda_path = None

    print('CUDA_PATH:', _cuda_path)
    
    return _cuda_path

ext_modules = [
    Extension(
        'cupy_chol',
        ['cupychol/cupy_chol.cpp'],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            get_cuda_path() + '/include'
        ],
        libraries=['cusolver', 'cusparse', 'cudart'],
        library_dirs=[get_cuda_path() + '/lib64'],
        language='c++'
    ),
]

setup(
    name='cupy_chol',
    version='0.1.4',
    author='Amin Heyrani Nobari',
    author_email='',
    description='cupychol: Solve linear systems using Cholesky decomposition with CuPy arrays on GPU.',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.5.0', 'cupy>=8.0.0', 'numpy>=1.18.0', 'scipy>=1.4.0'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)
