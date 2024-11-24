#!/bin/bash
#SBATCH --job-name=knn_comparison
#SBATCH --output=knn_comparison-mpi.out
#SBATCH --error=knn_comparison-mpi.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --partition=batch
#SBATCH --qos=normal

# Load necessary modules
module load lang/Python/3.8.6-GCCcore-10.2.0
module load mpi/OpenMPI/4.0.5-GCC-10.2.0

# Run the Python script
mpirun -np 64 python knnmpi.py

