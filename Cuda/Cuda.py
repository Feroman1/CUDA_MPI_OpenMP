import numpy as np
from numba import cuda
import time
import matplotlib.pyplot as plt

@cuda.jit
def matrix_multiply(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

def main():
    matrix_size = 6000
    num_iterations = 5
    threads_list = [1, 2, 4, 8, 16, 32, 64]

    timings = []

    for _ in threads_list:
        A = np.random.rand(matrix_size, matrix_size)
        B = np.random.rand(matrix_size, matrix_size)
        C = np.zeros_like(A)

        # Copy data to device
        d_A = cuda.to_device(A)
        d_B = cuda.to_device(B)
        d_C = cuda.to_device(C)

        # Set up the grid and block dimensions
        threads_per_block = (16, 16)

        # Подгоните размер блока так, чтобы общее количество потоков не превышало максимальное
        blocks_per_grid = (
            (matrix_size + threads_per_block[0] - 1) // threads_per_block[0],
            (matrix_size + threads_per_block[1] - 1) // threads_per_block[1]
        )

        # Warm-up
        matrix_multiply[blocks_per_grid, threads_per_block](d_A, d_B, d_C)

        # Measure time
        start_time = time.time()

        for _ in range(num_iterations):
            matrix_multiply[blocks_per_grid, threads_per_block](d_A, d_B, d_C)

        cuda.synchronize()
        elapsed_time = (time.time() - start_time) / num_iterations
        timings.append(elapsed_time)

    # Plot the results
    plt.plot(threads_list, timings, marker='o')
    plt.title('Matrix Multiplication with CUDA')
    plt.xlabel('Threads per Block')
    plt.ylabel('Average Time per Iteration (s)')
    plt.show()

if __name__ == "__main__":
    main()