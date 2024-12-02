#!/bin/bash
#SBATCH --job-name=convolution_comparison
#SBATCH --output=convolution_comparison.out
#SBATCH --error=convolution_comparison.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=30
#SBATCH --partition=batch
#SBATCH --qos=normal

# Load necessary modules
module purge
module load lang/Python/3.8.6-GCCcore-10.2.0
module load mpi/OpenMPI/4.0.5-GCC-10.2.0

# Ensure the correct MPI and Python paths are used
export PATH=/opt/apps/resif/aion/2020b/epyc/software/OpenMPI/4.0.5-GCC-10.2.0/bin:$PATH
export PYTHONPATH=~/.local/lib/python3.8/site-packages:$PYTHONPATH

# Install required Python libraries
pip install --user --upgrade --force-reinstall --no-cache-dir numpy matplotlib mpi4py tensorflow absl-py jax jaxlib

# Verify mpi4py installation
echo "Verifying mpi4py installation..."
python3 -c "import mpi4py" || { echo "mpi4py not installed or not in PYTHONPATH"; exit 1; }

# Test mpi4py with mpirun
echo "Testing mpi4py with mpirun..."
mpirun -np 2 python3 -c "from mpi4py import MPI; print(MPI.Get_version())" || { echo "mpi4py test with mpirun failed"; exit 1; }

# Run the testbench with MPI
echo "Running Testbench with MPI..."
mpirun -np 2 python3 parallel_tb.py

