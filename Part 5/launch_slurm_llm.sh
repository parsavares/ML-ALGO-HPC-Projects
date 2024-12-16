#!/bin/sh -l
#SBATCH --partition=gpu
#SBATCH --gpus-per-node 2 # <---- number of gpus per node
#SBATCH -c 24
#SBATCH -t 40
#SBATCH -N 1  # <------ number of nodes. Keep it to '1' because it does not work. "AttributeError: 'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'".
#SBATCH --export=ALL

# get host name
hosts_file="hosts.txt"
scontrol show hostname $SLURM_JOB_NODELIST > $hosts_file

# Collect public key and accept them
while read -r node; do
    ssh-keyscan "$node" >> ~/.ssh/known_hosts
done < "$hosts_file"

# Create the host file containing node names and the number of GPUs
function makehostfile() {
perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
$slots=4 if $slots==0;
@nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
print map { "$b$_ slots=$slots\n" } @nodes'
}
makehostfile > hostfile


source /work/projects/ulhpc-tutorials/PS10-Horovod/env_ds.sh


# Log system info
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $(cat $hosts_file)"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"

# Run DeepSpeed jobs (1 GPU first, then 2 GPUs)
echo "===== Running 1-GPU Training =====" >> 1_gpu_output.txt
deepspeed --num_gpus 1 --num_nodes 1 --hostfile hostfile ./LLM.py >> 1_gpu_output4.txt

echo "===== Running 2-GPU Training =====" >> 2_gpu_output.txt
deepspeed --num_gpus 2 --num_nodes 1 --hostfile hostfile ./LLM.py >> 2_gpu_output4.txt

# Print job completion message
echo "Job completed at $(date)"


