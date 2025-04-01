#!/usr/bin/env python3
"""
OpenMP와 Cython을 사용한 병렬 계산 예제 테스트 스크립트
"""

import numpy as np
import time
import sys
import os

# 모듈이 컴파일되어 있는지 확인
try:
    import parallel_sum as ps
    import parallel_matmul as pm
except ImportError as e:
    print(f"Cython 모듈을 불러오는 중 오류 발생: {e}")
    print("다음 명령을 실행하세요: python setup.py build_ext --inplace")
    sys.exit(1)

def test_array_sum():
    """
    배열 합산 예제를 테스트합니다.
    """
    print("\n" + "="*50)
    print("배열 합산 테스트")
    print("="*50)
    
    # 다양한 배열 크기로 테스트
    for size in [1000000, 10000000, 50000000]:
        print(f"\n배열 크기: {size:,}")
        
        # 테스트용 배열 생성
        arr = ps.create_work_array(size)
        
        # 다양한 스레드 수로 테스트
        for num_threads in [1, 2, 4, 8]:
            serial_time, parallel_time, serial_sum, parallel_sum_result = ps.benchmark_sum(arr, num_threads)
            
            # 결과 출력
            speedup = serial_time / parallel_time if parallel_time > 0 else 0
            print(f"스레드 수: {num_threads}")
            print(f"  순차 실행 시간: {serial_time:.6f}초")
            print(f"  병렬 실행 시간: {parallel_time:.6f}초")
            print(f"  속도 향상: {speedup:.2f}x")
            
            # 정확도 확인
            if abs(serial_sum - parallel_sum_result) > 1e-10:
                print(f"  경고: 결과가 다릅니다. 차이: {abs(serial_sum - parallel_sum_result)}")
            else:
                print("  정확도: OK")

def test_matrix_multiplication():
    """
    행렬 곱셈 예제를 테스트합니다.
    """
    print("\n" + "="*50)
    print("행렬 곱셈 테스트")
    print("="*50)
    
    # 다양한 행렬 크기로 테스트
    for size in [200, 500, 1000]:
        print(f"\n행렬 크기: {size}x{size}")
        
        # 테스트용 행렬 생성
        A, B = pm.create_matrices(size)
        
        # 다양한 스레드 수로 테스트
        for num_threads in [1, 2, 4, 8]:
            serial_time, parallel_time, opt_time = pm.benchmark_matmul(A, B, num_threads)
            
            # 결과 출력
            speedup1 = serial_time / parallel_time if parallel_time > 0 else 0
            speedup2 = serial_time / opt_time if opt_time > 0 else 0
            
            print(f"스레드 수: {num_threads}")
            print(f"  순차 실행 시간: {serial_time:.6f}초")
            print(f"  기본 병렬 실행 시간: {parallel_time:.6f}초 (속도 향상: {speedup1:.2f}x)")
            print(f"  최적화 병렬 실행 시간: {opt_time:.6f}초 (속도 향상: {speedup2:.2f}x)")

def main():
    """
    메인 테스트 함수
    """
    print("OpenMP 및 Cython 병렬 처리 테스트")
    
    # 현재 시스템의 스레드 수 확인
    current_threads = ps.get_num_threads()
    print(f"시스템에서 사용 가능한 OpenMP 스레드 수: {current_threads}")
    
    # 배열 합산 테스트
    test_array_sum()
    
    # 행렬 곱셈 테스트
    test_matrix_multiplication()

if __name__ == "__main__":
    main() 