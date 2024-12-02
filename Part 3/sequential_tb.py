import numpy as np
import time
from sequential import convolution_2d as seq_convolution_2d  # Sequential version

# Parameters for benchmarking
num_runs = 30  # Number of iterations for benchmarking

# Dataset loading
from tensorflow.keras.datasets import mnist
(x_train, _), _ = mnist.load_data()
x = x_train[0].astype(np.float32) / 255.0  # Use the first image and normalize it

# Kernel initialization
kernel = np.array([[0.01, 0.0, 0.0],
                   [-1.0, 0.0, 1.0],
                   [0.0, 0.0, 0.0]])  # Same kernel as in the parallel script

results = []

# Benchmark loop
real_times = []
cpu_times = []

for _ in range(num_runs):
    # Measure real time
    start_time = time.time()

    # Perform sequential convolution
    start_cpu_time = time.process_time()
    output = seq_convolution_2d(x, kernel)
    end_cpu_time = time.process_time()

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

results.append({
    "real_time_avg": real_time_avg,
    "real_time_std": real_time_std,
    "cpu_time_avg": cpu_time_avg,
    "cpu_time_std": cpu_time_std
})

# Display results
print("Sequential Benchmark Results:")
print(f"{'Real Time Avg (s)':<20} {'Real Time Std (s)':<20} {'CPU Time Avg (s)':<20} {'CPU Time Std (s)':<20}")
for res in results:
    print(f"{res['real_time_avg']:<20.6f} {res['real_time_std']:<20.6f} {res['cpu_time_avg']:<20.6f} {res['cpu_time_std']:<20.6f}")
