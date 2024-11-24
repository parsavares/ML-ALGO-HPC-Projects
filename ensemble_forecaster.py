# ensemble_forecaster.py
'''
    Parsa VARES

    Tom WALTER

    Luc PEREIRA CARDOSO
'''

import jax  # For installation: pip install jax jaxlib
import jax.numpy as jnp
from mpi4py import MPI  # For parallel computing

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Rank of the current process
size = comm.Get_size()  # Total number of processes

# Initialize data
X = jnp.array([[0.1, 0.4], [0.1, 0.5], [0.1, 0.6]])  # Input example
y = jnp.array([[0.1, 0.7]])  # Expected output
W = jnp.array([[0., 1., 0., 1., 0., 1.], [0., 1., 0., 1., 0., 1.]])  # Neural network parameters
b = jnp.array([0.1])  # Neural network bias

################################
# DEFINITION OF THE FORECASTER #
################################

def forecast_1step(X: jnp.array, W: jnp.array, b: jnp.array) -> jnp.array:
    """
    Predicts the next step using the neural network.
    """
    X_flatten = X.flatten()
    y_next = jnp.dot(W, X_flatten) + b
    return y_next

def forecast(horizon: int, X: jnp.array, W: jnp.array, b: jnp.array) -> jnp.array:
    """
    Predicts future values over a specified horizon.
    """
    result = []

    # Loop over 'horizon' to predict future values
    for t in range(horizon):
        X_flatten = X.flatten()  # Flatten the window for dot product

        # Get the next prediction
        y_next = forecast_1step(X_flatten, W, b)

        # Update X by shifting rows and adding the new prediction in the last row
        X = jnp.roll(X, shift=-1, axis=0)  # Shift rows upwards
        X = X.at[-1].set(y_next)  # Update the last row with the new prediction

        # Append the prediction to results
        result.append(y_next)

    return jnp.array(result)

def forecast_1step_with_loss(params: tuple, X: jnp.array, y: jnp.array) -> float:
    """
    Computes the loss between the predicted and expected output.
    """
    W, b = params
    y_next = forecast_1step(X, W, b)
    return jnp.sum((y_next - y) ** 2)

####################################
# DEFINITION OF THE TRAINING LOOP  #
####################################

grad = jax.grad(forecast_1step_with_loss)

def training_loop(grad: callable, num_epochs: int, W: jnp.array, b: jnp.array, X: jnp.array, y: jnp.array) -> tuple:
    """
    Trains the neural network parameters using gradient descent.
    """
    for i in range(num_epochs):
        delta = grad((W, b), X, y)
        W -= 0.1 * delta[0]
        b -= 0.1 * delta[1]
    return W, b

###########################
# Ensemble of Forecasters #
###########################

# Number of forecasters to create
num_forecaster = 600  # Adjusted to be divisible by the number of processes
noise_std = 0.1  # Standard deviation for noise to ensure different initial conditions
horizon = 5  # Forecast horizon
num_epochs = 20  # Number of training epochs

# Each process will collect its own forecasting results
aggregated_forecasting_local = []

# Distribute the work among processes
for i in range(rank, num_forecaster, size):
    # Generate a unique random seed for each forecaster
    key = jax.random.PRNGKey(i)
    W_noise = jax.random.normal(key, W.shape) * noise_std
    b_noise = jax.random.normal(key, b.shape) * noise_std

    # Initialize weights and biases with noise
    W_init = W + W_noise
    b_init = b + b_noise

    # Train the neural network
    W_trained, b_trained = training_loop(grad, num_epochs, W_init, b_init, X, y)

    # Make predictions
    y_predicted = forecast(horizon, X, W_trained, b_trained)

    # Append the prediction to the local list
    aggregated_forecasting_local.append(y_predicted)

# Gather all local forecasts to the root process
aggregated_forecasting = comm.gather(aggregated_forecasting_local, root=0)

#########################
# Statistical Analysis  #
#########################

if rank == 0:
    # Flatten the list of forecasts
    aggregated_forecasting_flat = [forecast for sublist in aggregated_forecasting for forecast in sublist]
    aggregated_forecasting_array = jnp.array(aggregated_forecasting_flat)

    # Compute statistics
    mean_forecast = jnp.mean(aggregated_forecasting_array, axis=0)
    median_forecast = jnp.median(aggregated_forecasting_array, axis=0)
    std_forecast = jnp.std(aggregated_forecasting_array, axis=0)
    percentile_5 = jnp.percentile(aggregated_forecasting_array, 5, axis=0)
    percentile_95 = jnp.percentile(aggregated_forecasting_array, 95, axis=0)

    # Display the statistics
    print("Statistical Analysis of the Ensemble Forecasts:")
    print(f"Mean forecast:\n{mean_forecast}")
    print(f"Median forecast:\n{median_forecast}")
    print(f"Standard deviation:\n{std_forecast}")
    print(f"5th percentile:\n{percentile_5}")
    print(f"95th percentile:\n{percentile_95}")
