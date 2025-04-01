# OpenMP 및 Cython을 이용한 병렬 처리 예제

이 프로젝트는 OpenMP(Open Multi-Processing)와 Cython을 사용하여 Python에서 병렬 처리를 구현하는 방법을 보여주는 예제입니다.

## 개요

OpenMP는 공유 메모리 병렬 처리를 위한 API로, C, C++ 및 Fortran과 같은 컴파일 언어에서 널리 사용됩니다. Cython은 Python 코드를 C로 변환하여 성능을 향상시키는 도구로, OpenMP와 함께 사용하면 Python에서도 효율적인 병렬 처리를 구현할 수 있습니다.

이 프로젝트에는 두 가지 주요 예제가 포함되어 있습니다:

1. **배열 합 계산**: 큰 배열의 요소들을 병렬로 합산하는 예제
2. **행렬 곱셈**: 두 행렬의 곱을 병렬로 계산하는 예제

## 필요 조건

- Python 3.6 이상
- Cython (`pip install cython`)
- NumPy (`pip install numpy`)
- C/C++ 컴파일러 (GCC 권장)
- OpenMP가 지원되는 환경

## 설치 및 컴파일

1. 저장소를 클론하거나 다운로드합니다.
2. 필요한 라이브러리를 설치합니다:
   ```
   pip install -r requirements.txt
   ```
3. Cython 모듈을 컴파일합니다:
   ```
   python setup.py build_ext --inplace
   ```

## 실행 방법

컴파일이 완료되면 다음 명령으로 테스트 스크립트를 실행할 수 있습니다:

```
python test_openmp.py
```

이 스크립트는 다양한 크기의 배열과 행렬에 대해 순차 실행과 병렬 실행의 성능을 비교합니다.

## 예제 설명

### 1. 배열 합 계산 (`parallel_sum.pyx`)

이 예제는 대규모 배열의 모든 요소를 합산하는 작업을 보여줍니다:

- `sum_serial()`: 순차적으로 배열 요소를 합산
- `sum_parallel()`: OpenMP를 사용하여 병렬로 배열 요소를 합산
- `benchmark_sum()`: 두 방법의 성능을 비교

주요 OpenMP 기능:

- `prange`: 병렬 for 루프
- `reduction`: 각 스레드의 부분 결과를 자동으로 결합

### 2. 행렬 곱셈 (`parallel_matmul.pyx`)

이 예제는 행렬 곱셈을 구현하는 세 가지 방법을 보여줍니다:

- `matmul_serial()`: 기본적인 3중 루프를 사용한 순차 구현
- `matmul_parallel()`: OpenMP를 사용한 병렬 구현
- `matmul_parallel_optimized()`: 캐시 지역성을 최적화한 병렬 구현
- `benchmark_matmul()`: 세 가지 방법의 성능을 비교

주요 OpenMP 기능:

- 병렬 영역 (`parallel()`)
- 작업 스케줄링 (`schedule='dynamic'`)
- 청크 크기 조정 (`chunksize=1`)

## OpenMP 주요 개념

1. **병렬 영역 (Parallel Region)**

   ```cython
   with nogil, parallel():
       # 병렬로 실행될 코드
   ```

2. **작업 분배 (Work Sharing)**

   ```cython
   for i in prange(n, schedule='static'):
       # 병렬로 실행될 루프
   ```

3. **리덕션 (Reduction)**

   ```cython
   for i in prange(n, reduction='+:total'):
       total += arr[i]
   ```

4. **스케줄링 (Scheduling)**

   - `static`: 루프 반복을 균등하게 분배
   - `dynamic`: 각 스레드가 작업을 완료하면 새 작업을 할당
   - `guided`: 청크 크기가 점차 감소하는 동적 할당

5. **스레드 관리**
   ```cython
   omp_set_num_threads(num)  # 스레드 수 설정
   omp_get_num_threads()  # 현재 스레드 수 확인
   omp_get_thread_num()  # 현재 스레드 ID 확인
   ```

## 성능 최적화 팁

1. **작업 크기**: 병렬화할 작업은 충분히 커야 오버헤드보다 이득이 큽니다.
2. **작업 분배**: 불균등한 작업에는 `dynamic` 스케줄링이 유리합니다.
3. **메모리 접근**: 캐시 지역성을 고려한 데이터 접근 패턴이 중요합니다.
4. **스레드 수**: 항상 코어 수에 맞춰 최적의 스레드 수를 찾아야 합니다.
5. **동기화 최소화**: 스레드 간 동기화는 최소화해야 합니다.

## 라이센스

MIT 라이센스

## 참고 자료

- [OpenMP 공식 웹사이트](https://www.openmp.org/)
- [Cython 문서](https://cython.readthedocs.io/)
- [Cython 병렬화 문서](https://cython.readthedocs.io/en/latest/src/userguide/parallelism.html)
