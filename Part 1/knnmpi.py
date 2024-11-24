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
    num_runs = 30
    k = 2
    test_size = 1000

    # Lists to collect timing data for benchmarks
    real_times = []
    cpu_times = []

    # Generate data on rank 0
    if rank == 0:
        X_train = np.random.rand(rows*cols).reshape((rows,cols))
        y_train = np.random.randint(2, size=rows)
        X_test = np.random.randint(rows, size=test_size)
    else:
        X_train = None
        y_train = None
        X_test = None

    # Broadcast data to all ranks
    X_train = comm.bcast(X_train, root=0)
    y_train = comm.bcast(y_train, root=0)
    X_test = comm.bcast(X_test, root=0)
    comm.Barrier()

    # Initialize and fit the custom KNN classifier
    knn = KNNClassifier(k=k)
    knn.fit(X_train, y_train)
    comm.Barrier()

    # Divide X_test into chunks for each rank
    if rank == 0:
        X_test_chunks = np.array_split(X_test, size, axis=0)
    else:
        X_test_chunks = None

    # Scatter X_test chunks to each rank
    local_X_test = comm.scatter(X_test_chunks, root=0)
    local_X_train_subset = X_train[local_X_test]

    # Perform multiple runs for benchmarking
    for run in range(num_runs):
        comm.Barrier()

        # Start timing for real and CPU time
        start_real_time = time.time()
        start_cpu_time = time.process_time()

        # Each rank makes predictions on its local chunk of X_test
        local_predictions = knn.predict(local_X_train_subset)
        comm.Barrier()

        # Gather all local predictions at the root rank
        all_predictions = comm.gather(local_predictions, root=0)

        # End timing for real and CPU time
        end_real_time = time.time()
        end_cpu_time = time.process_time()

        if rank == 0:
            real_times.append(end_real_time - start_real_time)
            cpu_times.append(end_cpu_time - start_cpu_time)

            # Combine and evaluate results only on the root rank
            all_predictions = np.concatenate(all_predictions)
            correct = np.sum(all_predictions == y_train[X_test])
            print(f"[Run {run + 1}] Correct predictions: {correct}/{test_size}")

    # Calculate and display benchmark statistics on the root rank
    if rank == 0:
        real_time_avg = np.mean(real_times)
        real_time_std = np.std(real_times)
        cpu_time_avg = np.mean(cpu_times)
        cpu_time_std = np.std(cpu_times)

        print("\nBenchmark Results (30 runs):")
        print(f"Real Time - Average: {real_time_avg:.4f} s, Std Dev: {real_time_std:.4f} s")
        print(f"CPU Time  - Average: {cpu_time_avg:.4f} s, Std Dev: {cpu_time_std:.4f} s")

if __name__ == "__main__":
    main()

