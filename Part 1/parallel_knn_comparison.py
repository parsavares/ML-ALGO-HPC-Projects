from KNNClassifier import KNNClassifier
import numpy as np
from multiprocessing import Pool, cpu_count
import time
import timeit
from tqdm import tqdm

# Sequential KNN Regressor class (used for benchmarking against the parallel version)
class SequentialKNNRegressor(KNNClassifier):
    def _predict(self, x):
        # Compute Euclidean distances from test point 'x' to all training points
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        # Select the indices of the k closest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Retrieve the target values of these k closest neighbors
        k_nearest_values = [self.y_train[i] for i in k_indices]
        # For regression, return the mean value of the k nearest neighbors
        return np.mean(k_nearest_values)

    def predict(self, X):
        # Sequentially compute predictions for each test point in X
        return np.array([self._predict(x) for x in tqdm(X, desc="Sequential KNN")])

# Parallel KNN Regressor class (distributes predictions across multiple CPU cores)
class ParallelKNNRegressor(KNNClassifier):
    def _predict(self, x):
        # Compute Euclidean distances from test point 'x' to all training points
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        # Select the indices of the k closest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Retrieve the target values of these k closest neighbors
        k_nearest_values = [self.y_train[i] for i in k_indices]
        # For regression, return the mean value of the k nearest neighbors
        return np.mean(k_nearest_values)

    def predict(self, X):
        # Use multiprocessing to compute predictions in parallel for each test point in X
        with Pool(cpu_count()) as pool:
            y_pred = list(tqdm(pool.imap(self._predict, X), total=len(X), desc="Parallel KNN"))
        return np.array(y_pred)

# Function to benchmark and compare Sequential and Parallel KNN implementations
def benchmark_comparison(sequential_knn, parallel_knn, X_train, y_train, X_test, runs=30):
    # Lists to store real-time and CPU times for each run of both versions
    seq_real_times = []
    seq_cpu_times = []
    par_real_times = []
    par_cpu_times = []

    # Benchmark Sequential KNN
    for _ in tqdm(range(runs), desc="Benchmarking Sequential KNN"):
        start_real_time = time.time()  # Real time start
        start_cpu_time = timeit.default_timer()  # CPU time start
        
        seq_predictions = sequential_knn.predict(X_train[X_test])  # Run predictions
        
        end_real_time = time.time()  # Real time end
        end_cpu_time = timeit.default_timer()  # CPU time end
        
        # Append real and CPU times for this run
        seq_real_times.append(end_real_time - start_real_time)
        seq_cpu_times.append(end_cpu_time - start_cpu_time)

    # Benchmark Parallel KNN
    for _ in tqdm(range(runs), desc="Benchmarking Parallel KNN"):
        start_real_time = time.time()  # Real time start
        start_cpu_time = timeit.default_timer()  # CPU time start
        
        par_predictions = parallel_knn.predict(X_train[X_test])  # Run predictions in parallel
        
        end_real_time = time.time()  # Real time end
        end_cpu_time = timeit.default_timer()  # CPU time end
        
        # Append real and CPU times for this run
        par_real_times.append(end_real_time - start_real_time)
        par_cpu_times.append(end_cpu_time - start_cpu_time)

    # Calculate average and standard deviation for Sequential KNN times
    avg_seq_real_time = np.mean(seq_real_times)
    std_seq_real_time = np.std(seq_real_times)
    avg_seq_cpu_time = np.mean(seq_cpu_times)
    std_seq_cpu_time = np.std(seq_cpu_times)

    # Calculate average and standard deviation for Parallel KNN times
    avg_par_real_time = np.mean(par_real_times)
    std_par_real_time = np.std(par_real_times)
    avg_par_cpu_time = np.mean(par_cpu_times)
    std_par_cpu_time = np.std(par_cpu_times)

    # Compute speed-up as ratio of sequential time to parallel time
    speedup_real_time = avg_seq_real_time / avg_par_real_time
    speedup_cpu_time = avg_seq_cpu_time / avg_par_cpu_time

    # Print out benchmark results for comparison
    print("Sequential KNN (Average Real Time): {:.4f}s ± {:.4f}s".format(avg_seq_real_time, std_seq_real_time))
    print("Sequential KNN (Average CPU Time): {:.4f}s ± {:.4f}s".format(avg_seq_cpu_time, std_seq_cpu_time))
    print("Parallel KNN (Average Real Time): {:.4f}s ± {:.4f}s".format(avg_par_real_time, std_par_real_time))
    print("Parallel KNN (Average CPU Time): {:.4f}s ± {:.4f}s".format(avg_par_cpu_time, std_par_cpu_time))
    print("Speed-up in Real Time: {:.2f}x".format(speedup_real_time))
    print("Speed-up in CPU Time: {:.2f}x".format(speedup_cpu_time))

if __name__ == "__main__":
    # Data preparation for benchmarking
    rows = 100000  # Number of rows in training data
    cols = 500  # Number of columns (features) in training data
    np.random.seed(699)  # Seed for reproducibility
    X_train = np.random.rand(rows * cols).reshape((rows, cols))  # Random training data
    y_train = np.random.rand(rows)  # Random continuous values for regression targets

    # Instantiate both Sequential and Parallel KNN models with k=2 neighbors
    sequential_knn = SequentialKNNRegressor(k=2)
    parallel_knn = ParallelKNNRegressor(k=2)
    
    # Fit both models on the training data
    sequential_knn.fit(X_train, y_train)
    parallel_knn.fit(X_train, y_train)

    # Generate random indices for testing (subset of training data for simplicity)
    test_size = 1000  # Number of test samples
    X_test = np.random.randint(rows, size=test_size)

    # Execute the benchmark comparison between sequential and parallel implementations
    benchmark_comparison(sequential_knn, parallel_knn, X_train, y_train, X_test)
