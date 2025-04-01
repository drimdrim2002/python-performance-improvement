from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# OpenMP 지원을 위한 컴파일러 옵션 설정
ext_modules = [
    Extension(
        "parallel_sum",
        ["parallel_sum.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "parallel_matmul",
        ["parallel_matmul.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[np.get_include()]
    )
]

setup(
    name="cython_openmp_examples",
    ext_modules=cythonize(ext_modules),
    include_dirs=[np.get_include()],
    requires=["numpy"]
) 