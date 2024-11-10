from mpi4py import MPI
import numpy as np
from knn_code.KNNClassifier import KNNClassifier
import time

def main():
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parameters and data setup
    rows, cols = 100000, 500
    np.random.seed(699)   
    k = 2
    test_size = 1000

    # Timing data for a single benchmark run
    start_real_time = time.time()
    start_cpu_time = time.process_time()

    # Generate data (each rank creates its own full dataset)
    X_train = np.random.rand(rows*cols).reshape((rows,cols))
    y_train = np.random.randint(2, size=rows)
    X_test = np.random.randint(rows, size=test_size)

    # Initialize and fit the custom KNN classifier on each rank
    knn = KNNClassifier(k=k)
    knn.fit(X_train, y_train)

    # Make predictions on the entire X_test
    predictions = knn.predict(X_train[X_test])

    # End timing for real and CPU time
    end_real_time = time.time()
    end_cpu_time = time.process_time()

    # Calculate correctness for this run
    correct = np.sum(predictions == y_train[X_test])

    # Calculate elapsed times
    real_time_elapsed = end_real_time - start_real_time
    cpu_time_elapsed = end_cpu_time - start_cpu_time

    # Each rank sends its results to the root
    result = (real_time_elapsed, cpu_time_elapsed, correct)
    all_results = comm.gather(result, root=0)

    # Root rank calculates the overall statistics
    if rank == 0:
        real_times = [res[0] for res in all_results]
        cpu_times = [res[1] for res in all_results]
        correct_counts = [res[2] for res in all_results]

        # Calculate statistics
        real_time_avg = np.mean(real_times)
        real_time_std = np.std(real_times)
        cpu_time_avg = np.mean(cpu_times)
        cpu_time_std = np.std(cpu_times)
        total_correct = np.sum(correct_counts)
        avg_correct = np.mean(correct_counts)

        # Display results
        print("\nTotal Benchmark Results (30 runs):")
        print(f"Real Time - Average: {real_time_avg:.4f} s, Std Dev: {real_time_std:.4f} s")
        print(f"CPU Time  - Average: {cpu_time_avg:.4f} s, Std Dev: {cpu_time_std:.4f} s")
        print(f"Total Correct Predictions: {total_correct} / {test_size * size}")
        print(f"Average Correct Predictions per Run: {avg_correct:.2f} / {test_size}")

if __name__ == "__main__":
    main()

