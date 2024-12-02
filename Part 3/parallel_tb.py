import numpy as np
import time
from mpi4py import MPI
from parallel import convolution_2d as par_convolution_2d  # Updated parallel version  # Parallel version


# MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters for benchmarking
chunk_sizes = [28 // size, 14, 7]  # Example: Full row split, half row split, finer splits
num_runs = 30  # Number of iterations for benchmarking

# Dataset loading
if rank == 0:
    from tensorflow.keras.datasets import mnist
    (x_train, _), _ = mnist.load_data()
    x = x_train[0].astype(np.float32) / 255.0
else:
    x = None

# Broadcast the data to all processes
x = comm.bcast(x, root=0)

# Kernel initialization
kernel = np.array([[0.01, 0.0, 0.0],
                   [-1.0, 0.0, 1.0],
                   [0.0, 0.0, 0.0]])  # Same as in the parallel script

results = []

# Benchmark loop
for chunk_size in chunk_sizes:
    real_times = []
    cpu_times = []

    for _ in range(num_runs):
        # Measure real time
        start_time = time.time()

        # Local data allocation
        local_start = rank * chunk_size
        local_end = local_start + chunk_size if rank < size - 1 else x.shape[0]
        local_x = x[local_start:local_end]

        # Perform parallel convolution
        start_cpu_time = time.process_time()
        local_output = par_convolution_2d(local_x, kernel)
        end_cpu_time = time.process_time()

        # Gather results at the root
        if rank == 0:
            full_output = np.zeros_like(x)
        else:
            full_output = None
        comm.Gather(np.array(local_output, dtype=np.float32), full_output, root=0)

        # End real-time measurement
        end_time = time.time()

        # Append times
        real_times.append(end_time - start_time)
        cpu_times.append(end_cpu_time - start_cpu_time)

    # Calculate average and standard deviation
    real_time_avg = np.mean(real_times)
    real_time_std = np.std(real_times)
    cpu_time_avg = np.mean(cpu_times)
    cpu_time_std = np.std(cpu_times)

    if rank == 0:
        results.append({
            "chunk_size": chunk_size,
            "real_time_avg": real_time_avg,
            "real_time_std": real_time_std,
            "cpu_time_avg": cpu_time_avg,
            "cpu_time_std": cpu_time_std
        })

# Display results (only on root)
if rank == 0:
    print("Benchmark Results:")
    print(f"{'Chunk Size':<15} {'Real Time Avg (s)':<20} {'Real Time Std (s)':<20} {'CPU Time Avg (s)':<20} {'CPU Time Std (s)':<20}")
    for res in results:
        print(f"{res['chunk_size']:<15} {res['real_time_avg']:<20.6f} {res['real_time_std']:<20.6f} {res['cpu_time_avg']:<20.6f} {res['cpu_time_std']:<20.6f}")
