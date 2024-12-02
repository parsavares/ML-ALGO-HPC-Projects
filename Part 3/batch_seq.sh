#!/bin/bash
#SBATCH --job-name=convolution_comparison
#SBATCH --output=convolution_comparison.out
#SBATCH --error=convolution_comparison.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=batch
#SBATCH --qos=normal

# Load necessary modules
module purge
module load lang/Python/3.8.6-GCCcore-10.2.0

# Ensure the correct Python paths are used
export PYTHONPATH=~/.local/lib/python3.8/site-packages:$PYTHONPATH

# Install required Python libraries
pip install --user --upgrade --force-reinstall --no-cache-dir numpy matplotlib tensorflow absl-py jax jaxlib

# Run the testbench
echo "Running Sequential Convolution Testbench..."
python3 sequential_tb.py || { echo "Testbench execution failed"; exit 1; }

echo "Sequential Convolution Testbench completed successfully."
