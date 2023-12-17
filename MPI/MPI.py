import numpy as np
from mpi4py import MPI
import time
import matplotlib.pyplot as plt

def generate_matrix(rows, cols):
    return np.random.rand(rows, cols)

def parallel_matrix_multiply(matrix_A, matrix_B):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rows_A, cols_A = matrix_A.shape
    rows_B, cols_B = matrix_B.shape

    assert cols_A == rows_B, "Invalid matrix dimensions for multiplication"

    block_size = rows_A // size
    local_A = np.zeros((block_size, cols_A))
    comm.Scatter(matrix_A, local_A, root=0)

    local_C = np.zeros((block_size, cols_B))
    comm.Bcast(matrix_B, root=0)

    for i in range(block_size):
        for j in range(cols_B):
            local_C[i, j] = np.sum(local_A[i, :] * matrix_B[:, j])

    result = None
    if rank == 0:
        result = np.zeros((rows_A, cols_B))
    comm.Gather(local_C, result, root=0)

    return result

def main():
    matrix_size = 800  # Размер матриц (должен быть достаточно большим)
    num_runs = 5       # Количество запусков для усреднения времени выполнения

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("Matrix size:", matrix_size)

    # Генерация случайных матриц
    matrix_A = generate_matrix(matrix_size, matrix_size)
    matrix_B = generate_matrix(matrix_size, matrix_size)

    # Измерение времени выполнения для разного количества процессов
    num_processes = [1, 2, 4, 8, 16, 32, 64]
    execution_times = []

    for num_procs in num_processes:
        comm.Barrier()
        start_time = time.time()

        for _ in range(num_runs):
            parallel_matrix_multiply(matrix_A, matrix_B)

        comm.Barrier()
        end_time = time.time()
        average_time = (end_time - start_time) / num_runs
        execution_times.append(average_time)

        if rank == 0:
            print(f"Потоков: {num_procs}, Время выполнения: {average_time:.4f} сек")

    # Отображение графика
    if rank == 0:
        plt.plot(num_processes, execution_times, marker='o')
        plt.title("Matrix Multiplication with MPI")
        plt.xlabel("Number of Processes")
        plt.ylabel("Average Time (seconds)")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()