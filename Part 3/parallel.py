import numpy as np
import jax
import jax.numpy as jnp
from jax import grad
from mpi4py import MPI

# MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Utility function for logging
def log(msg):
    print(f"[Rank {rank}] {msg}")

# Convolution function using JAX
def convolution_2d(x, kernel):
    input_height, input_width = x.shape
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2

    # Pad input
    padded_x = jnp.pad(x, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    output_data = jnp.zeros_like(x)

    # Perform convolution
    for i in range(input_height):
        for j in range(input_width):
            region = padded_x[i:i + kernel_height, j:j + kernel_width]
            output_data = output_data.at[i, j].set(jnp.sum(region * kernel))

    return output_data

# Define loss function
def loss_fn(kernel, x, y_true):
    y_pred = convolution_2d(x, kernel)
    return jnp.mean((y_pred - y_true) ** 2)

# Data preparation
x_train, y_train = np.random.random((60000, 28, 28)), None
x = x_train[0]
kernel = jnp.array([[0.01, 0.0, 0.0],
                    [-1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0]])

log(f"Initial input shape: {x.shape}")
log(f"Kernel shape: {kernel.shape}")

# Padding to ensure divisibility
if x.shape[0] % size != 0:
    padding_rows = (size - (x.shape[0] % size)) % size
    x_padded = np.pad(x, ((0, padding_rows), (0, 0)), mode='constant')
else:
    padding_rows = 0  # No padding needed
    x_padded = x
y_true_padded = np.pad(x, ((0, padding_rows), (0, 0)), mode='constant')
log(f"Padding rows: {padding_rows}")

log(f"Padded input shape: {x_padded.shape}")
log(f"Padded target shape: {y_true_padded.shape}")

global_rows = x_padded.shape[0]
chunk_size = global_rows // size
chunk_sizes = [(chunk_size if rank < size - 1 else global_rows - rank * chunk_size) for rank in range(size)]

displacements = np.cumsum([0] + chunk_sizes[:-1])

# Local data distribution
local_start = rank * chunk_size
local_end = local_start + chunk_sizes[rank]

local_x = x_padded[local_start:local_end]
local_y_true = y_true_padded[local_start:local_end]

log(f"Local input shape: {local_x.shape}")
log(f"Local target shape: {local_y_true.shape}")

# Training loop
learning_rate = 0.01
num_iterations = 10

for i in range(num_iterations):
    # Compute predictions
    y_pred = convolution_2d(local_x, kernel)

    # Verify shapes match
    assert y_pred.shape == local_y_true.shape, (
        f"Shape mismatch: y_pred {y_pred.shape} vs local_y_true {local_y_true.shape}"
    )

    # Adjust loss_fn for local data
    def local_loss_fn(kernel, local_x, local_y_true):
        local_y_pred = convolution_2d(local_x, kernel)
        return jnp.mean((local_y_pred - local_y_true) ** 2)

    # Compute local loss and gradients
    local_loss = local_loss_fn(kernel, local_x, local_y_true)
    local_gradients = grad(local_loss_fn)(kernel, local_x, local_y_true)

    global_gradients = comm.allreduce(local_gradients, op=MPI.SUM)
    kernel -= learning_rate * global_gradients

    log(f"Iteration {i}: local loss: {local_loss}")

# Gathering results
if rank == 0:
    full_output = np.zeros((sum(chunk_sizes), local_x.shape[1]), dtype=np.float32)
else:
    full_output = None

# Prepare data to send
local_data = np.array(local_x, dtype=np.float32)

# Use MPI.Gather to collect local data into full_output on the root process
comm.Gather(local_data, full_output, root=0)

if rank == 0:
    full_output = full_output[:x.shape[0], :]
    log(f"Final output shape: {full_output.shape}")
