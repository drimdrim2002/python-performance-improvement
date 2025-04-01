# cython: language_level=3
# distutils: language=c++

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel
cimport openmp

# OpenMP 라이브러리를 사용하기 위한 cimport
cdef extern from "omp.h" nogil:
    int omp_get_thread_num()
    int omp_get_num_threads()
    void omp_set_num_threads(int num_threads)
    double omp_get_wtime()

def matmul_serial(np.ndarray[double, ndim=2] A, np.ndarray[double, ndim=2] B):
    """
    두 행렬의 곱을 순차적으로 계산합니다.
    
    Args:
        A: 첫 번째 행렬 (m x n)
        B: 두 번째 행렬 (n x p)
    
    Returns:
        행렬 곱 C = A * B (m x p)
    """
    cdef:
        int m = A.shape[0]
        int n = A.shape[1]
        int p = B.shape[1]
        np.ndarray[double, ndim=2] C = np.zeros((m, p), dtype=np.float64)
        int i, j, k
    
    # 전통적인 3중 루프 행렬 곱셈
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    
    return C

def matmul_parallel(np.ndarray[double, ndim=2] A, np.ndarray[double, ndim=2] B, int num_threads=0):
    """
    두 행렬의 곱을 병렬로 계산합니다.
    
    Args:
        A: 첫 번째 행렬 (m x n)
        B: 두 번째 행렬 (n x p)
        num_threads: 사용할 스레드 수 (0이면 시스템 기본값 사용)
    
    Returns:
        행렬 곱 C = A * B (m x p)
    """
    cdef:
        int m = A.shape[0]
        int n = A.shape[1]
        int p = B.shape[1]
        np.ndarray[double, ndim=2] C = np.zeros((m, p), dtype=np.float64)
        int i, j, k
    
    if num_threads > 0:
        omp_set_num_threads(num_threads)
    
    # 최외각 루프를 병렬화
    # schedule='dynamic'은 각 스레드가 작업 완료 후 새 작업을 할당받음
    # 이는 작업량이 불균형할 때 좋음
    with nogil:
        for i in prange(m, schedule='dynamic', chunksize=1):
            for j in range(p):
                for k in range(n):
                    C[i, j] += A[i, k] * B[k, j]
    
    return C

def matmul_parallel_optimized(np.ndarray[double, ndim=2] A, np.ndarray[double, ndim=2] B, int num_threads=0):
    """
    두 행렬의 곱을 최적화된 방법으로 병렬 계산합니다.
    
    캐시 지역성을 고려한 최적화와 병렬화를 모두 적용한 버전입니다.
    
    Args:
        A: 첫 번째 행렬 (m x n)
        B: 두 번째 행렬 (n x p)
        num_threads: 사용할 스레드 수 (0이면 시스템 기본값 사용)
    
    Returns:
        행렬 곱 C = A * B (m x p)
    """
    cdef:
        int m = A.shape[0]
        int n = A.shape[1]
        int p = B.shape[1]
        np.ndarray[double, ndim=2] C = np.zeros((m, p), dtype=np.float64)
        np.ndarray[double, ndim=2] B_T = np.transpose(B)  # B 전치
        int i, j, k
    
    if num_threads > 0:
        omp_set_num_threads(num_threads)
    
    # B를 전치하여 캐시 지역성 향상
    # 이제 내부 루프에서 연속된 메모리 접근이 가능
    with nogil:
        for i in prange(m, schedule='dynamic', chunksize=1):
            for j in range(p):
                # C[i, j]는 각 스레드가 독립적으로 계산
                C[i, j] = 0.0  # 초기화
                for k in range(n):
                    C[i, j] += A[i, k] * B_T[j, k]  # B_T[j, k] = B[k, j]
    
    return C

def benchmark_matmul(np.ndarray[double, ndim=2] A, np.ndarray[double, ndim=2] B, int num_threads=0):
    """
    다양한 행렬 곱셈 알고리즘의 성능을 비교합니다.
    
    Args:
        A: 첫 번째 행렬
        B: 두 번째 행렬
        num_threads: 병렬 버전에서 사용할 스레드 수
    
    Returns:
        (순차 시간, 병렬 시간, 최적화 병렬 시간) 튜플
    """
    cdef:
        double serial_start, serial_end, parallel_start, parallel_end
        double opt_start, opt_end
        double serial_time, parallel_time, opt_time
    
    # 순차 실행 시간 측정
    serial_start = omp_get_wtime()
    _ = matmul_serial(A, B)
    serial_end = omp_get_wtime()
    serial_time = serial_end - serial_start
    
    # 기본 병렬 실행 시간 측정
    parallel_start = omp_get_wtime()
    _ = matmul_parallel(A, B, num_threads)
    parallel_end = omp_get_wtime()
    parallel_time = parallel_end - parallel_start
    
    # 최적화된 병렬 실행 시간 측정
    opt_start = omp_get_wtime()
    _ = matmul_parallel_optimized(A, B, num_threads)
    opt_end = omp_get_wtime()
    opt_time = opt_end - opt_start
    
    return serial_time, parallel_time, opt_time

def create_matrices(int size):
    """
    테스트용 무작위 행렬을 생성합니다.
    
    Args:
        size: 행렬의 크기 (size x size)
    
    Returns:
        (A, B) 두 개의 무작위 행렬
    """
    A = np.random.random((size, size))
    B = np.random.random((size, size))
    return A, B 