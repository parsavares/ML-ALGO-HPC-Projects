#!/bin/sh -l
#SBATCH -c 4                # Requesting 4 CPU cores per node. The strategy employed is:
                            # 2 cores for each MPI rank: 1 core for the data loader, 1 core for the training loop.
#SBATCH -N 2                # Requesting 2 nodes for multi-node setup
#SBATCH -p gpu              # Note: Software stack (MPI/TensorFlow/PyTorch/Python) is compiled for GPU nodes.
                            # Even without requesting GPUs, the script requires GPU nodes.
#SBATCH --gpus-per-node 2 # This line is commented for now; it can be enabled if GPU usage is required.
#SBATCH -t 02:00:00               # Setting a runtime limit of 15 minutes
#SBATCH --export=ALL        # Exporting all environment variables to the job


source /work/projects/ulhpc-tutorials/PS10-Horovod/soft/miniconda/scripts/env.sh

#echo "*** TEST ***"
#mpirun -n 4 python test_horovod.py

echo "*** TENSORFLOW ***"
#mpirun -n 4 python ./tensorflow_v.py

echo "*** PYTORCH ***"
mpirun -n 4 python ./pytorch_v.py
