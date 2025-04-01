# cython: language_level=3
# distutils: language=c++

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel
from libc.stdlib cimport malloc, free
cimport openmp

# OpenMP 라이브러리를 사용하기 위한 cimport
cdef extern from "omp.h" nogil:
    int omp_get_thread_num()
    int omp_get_num_threads()
    void omp_set_num_threads(int num_threads)
    double omp_get_wtime()

def get_num_threads():
    """
    현재 설정된 OpenMP 스레드 수를 반환합니다.
    """
    # 병렬 영역에서 변수 공유 문제를 피하기 위해 간소화한 버전
    # 현재 사용 가능한 CPU 코어 수를 기반으로 OpenMP 스레드 기본값을 추정
    import multiprocessing
    return multiprocessing.cpu_count()

def set_num_threads(int num):
    """
    OpenMP 스레드 수를 설정합니다.
    """
    omp_set_num_threads(num)

def sum_serial(np.ndarray[double, ndim=1] arr):
    """
    배열 요소의 합을 순차적으로 계산합니다.
    """
    cdef:
        int i
        double total = 0.0
        int n = arr.shape[0]
    
    for i in range(n):
        total += arr[i]
    
    return total

def sum_parallel(np.ndarray[double, ndim=1] arr, int num_threads=0):
    """
    배열 요소의 합을 병렬로 계산합니다.
    
    Args:
        arr: 합산할 배열
        num_threads: 사용할 스레드 수 (0이면 시스템 기본값 사용)
    
    Returns:
        배열 요소의 합
    """
    cdef:
        int i
        double total = 0.0
        int n = arr.shape[0]
    
    if num_threads > 0:
        omp_set_num_threads(num_threads)
    
    # OpenMP parallel for 사용하여 합산 병렬화
    # 각 스레드가 부분 합을 계산한 후 결과를 합칩니다
    with nogil:
        # reduction 대신 명시적으로 처리
        for i in prange(n, schedule='static'):
            total += arr[i]
    
    return total

def benchmark_sum(np.ndarray[double, ndim=1] arr, int num_threads=0):
    """
    순차 실행과 병렬 실행의 성능을 비교합니다.
    
    Args:
        arr: 합산할 배열
        num_threads: 병렬 버전에서 사용할 스레드 수
    
    Returns:
        (순차 실행 시간, 병렬 실행 시간, 순차 합계, 병렬 합계) 튜플
    """
    cdef:
        double serial_start, serial_end, parallel_start, parallel_end
        double serial_time, parallel_time
        double serial_sum, parallel_sum
    
    # 순차 실행 시간 측정
    serial_start = omp_get_wtime()
    serial_sum = sum_serial(arr)
    serial_end = omp_get_wtime()
    serial_time = serial_end - serial_start
    
    # 병렬 실행 시간 측정
    parallel_start = omp_get_wtime()
    parallel_sum = sum_parallel(arr, num_threads)
    parallel_end = omp_get_wtime()
    parallel_time = parallel_end - parallel_start
    
    return serial_time, parallel_time, serial_sum, parallel_sum

def create_work_array(int size):
    """
    테스트용 배열을 생성합니다.
    """
    return np.random.random(size) 