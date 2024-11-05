## ML-ALGO-HPC
# Assignmet 1:
to get the err and output i used this but the time was not enough :
sbatch --job-name=knn_comparison --output=knn_comparison.out --error=knn_comparison.err --time=02:00:00 --nodes=1 --ntasks=1 --cpus-per-task=32 --partition=interactive --qos=debug --wrap="module load lang/Python/3.8.6-GCCcore-10.2.0 && source knn_env/bin/activate && python parallel_knn_comparison.py"
