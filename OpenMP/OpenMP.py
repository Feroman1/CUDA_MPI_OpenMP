import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import time
import os


# Функция для перемножения матриц с использованием OpenMP
@njit(parallel=True)
def matrix_multiply(A, B):
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in prange(A.shape[0]):
        for j in prange(B.shape[1]):
            for k in prange(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
    return C


# Функция для измерения времени выполнения
def measure_time(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    return result, end_time - start_time


# Задаем размер матрицы
matrix_size = 1600
num_runs = 5       # Количество запусков для усреднения времени выполнения
# Создаем случайные матрицы A и B
A = np.random.rand(matrix_size, matrix_size)
B = np.random.rand(matrix_size, matrix_size)

# Задаем количество потоков для тестирования
num_threads_values = [1, 2, 4, 8, 16, 32, 64]

# Измеряем время выполнения для разного количества потоков
execution_times = []

for num_threads in num_threads_values:
    os.environ['NUMBA_NUM_THREADS'] = str(num_threads)
    total_time = 0

    for _ in range(num_runs):
        _, elapsed_time = measure_time(matrix_multiply, A, B)
        total_time += elapsed_time

    average_time = total_time/num_runs
    execution_times.append(average_time)

    print(f"Потоков: {num_threads}, Время выполнения: {average_time:.4f} сек")

# Строим график зависимости времени выполнения от количества потоков
plt.plot(num_threads_values, execution_times, marker='o')
plt.title("Matrix Multiplication with OpenMP")
plt.xlabel("Number of Processes")
plt.ylabel("Average Time (seconds)")
plt.grid(True)
plt.show()