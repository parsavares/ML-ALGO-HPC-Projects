#!/bin/bash
#SBATCH --job-name=knn_comparison
#SBATCH --output=knn_comparison-seq_parallel.out
#SBATCH --error=knn_comparison-seq_parallel.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=30
#SBATCH --partition=batch
#SBATCH --qos=normal

# Load necessary modules
module load lang/Python/3.8.6-GCCcore-10.2.0
module load mpi/OpenMPI/4.0.5-GCC-10.2.0

# Run the Python script
mpirun -np 30 python seq_parallel.py
