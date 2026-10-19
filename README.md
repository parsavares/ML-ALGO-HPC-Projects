# Machine Learning Algorithms on HPC

**MPI parallelization · Distributed deep learning · GPU training experiments · Performance analysis**

Five projects connecting machine learning algorithms with the systems that run them: distributing KNN predictions, training ensembles with JAX, partitioning convolution workloads, training ResNet-50 with Horovod, and evaluating OPT-125M with DeepSpeed.

Developed collaboratively by **Parsa Vares, Tom Walter, and Luc Pereira Cardoso** for *Programming Machine Learning Algorithms for HPC* at the **University of Luxembourg**. The repository includes Python implementations, Slurm launchers, experiment logs, and four technical reports from 2024.

[Explore the projects](#projects-at-a-glance) · [Read the results](#recorded-benchmark-results) · [Run the code](#running-the-projects) · [Benchmark methodology](#benchmark-methodology-and-scope)

## Projects at a glance

For distributed AI training, start with **Parts 4 and 5**. For MPI programming and CPU performance analysis, start with **Parts 1–3**.

| Project | Implementation | Evidence |
| --- | --- | --- |
| **[1 · Parallel KNN](Part%201/)** | Prediction workload distributed across MPI ranks; sequential and parallel timing experiments | [Report](Part%201/ML.pdf) · [Benchmark logs](Part%201/out/) |
| **[2 · Ensemble forecasting](Part%202/)** | 600 JAX forecasters distributed with MPI; ensemble statistics aggregated on rank 0 | [Report](Part%202/ML2.pdf) · [Implementation](Part%202/ensemble_forecaster.py) |
| **[3 · Parallel convolution](Part%203/)** | JAX convolution, row partitioning, MPI result gathering, and process-count experiments | [Report](Part%203/ML_Part_3.pdf) · [Benchmark logs](Part%203/parallelOut/) |
| **[4 · Distributed ResNet-50](Part%204/)** | CIFAR-10 training in PyTorch and TensorFlow using Horovod and Slurm | [PyTorch](Part%204/pytorch_v.py) · [TensorFlow](Part%204/tensorflow_v.py) · [Launcher](Part%204/batch.sh) |
| **[5 · Language-model training](Part%205/)** | Hugging Face OPT-125M with DeepSpeed; GPU-count, batch-size, optimizer, and ZeRO-stage comparisons | [Report](Part%205/ML5.pdf) · [Implementation](Part%205/LLM.py) |

## Engineering focus

- **Parallel algorithm design:** identify expensive operations, partition work, and coordinate processes through broadcast, scatter, gather, and reduction.
- **Distributed AI training:** shard datasets, assign devices by local rank, synchronize model state, and integrate distributed optimizers.
- **HPC execution:** configure CPU/GPU resources, use MPI launchers, and submit workloads through Slurm on university clusters.
- **Experimental analysis:** compare wall-clock timings, summarize repeated measurements, and examine the trade-offs between workload size and parallel execution.

**Stack:** Python, NumPy, JAX, mpi4py/OpenMPI, PyTorch, TensorFlow/Keras, Horovod, Hugging Face Transformers & Datasets, DeepSpeed, Matplotlib, Bash, and Slurm.

## Project details

### 1. Parallel KNN prediction

The MPI implementation extends a provided KNN classifier by distributing the prediction workload. Rank 0 generates the data, broadcasts the training set, scatters query indices, and gathers predictions from the workers.

The experiment uses **100,000 training samples, 500 features, and 1,000 queries**, with 30 measurements for the reported sequential and parallel configurations. The repository also contains exploratory Python multiprocessing variants.

**Key files:** [MPI implementation](Part%201/knnmpi.py) · [Sequential benchmark](Part%201/seq_parallel.py) · [Base classifier](Part%201/knn_code/KNNClassifier.py)

### 2. JAX ensemble forecasting with MPI

An ensemble of **600 small forecasters** explores different parameter initializations on a synthetic forecasting example. Each forecaster is trained for 20 gradient-descent steps using JAX automatic differentiation and produces a five-step forecast.

Forecasters are assigned cyclically across MPI ranks. Rank 0 gathers predictions and computes the **mean, median, standard deviation, and 5th/95th percentiles**. The report documents execution with six MPI processes on the **Aion** cluster.

**Key file:** [Ensemble training and aggregation](Part%202/ensemble_forecaster.py)

### 3. Convolution and MPI workload partitioning

This project explores two-dimensional convolution implemented with JAX, gradient-based kernel updates, and an MNIST image experiment. The parallel benchmark partitions a 28-row image across **2, 4, 7, 14, and 28 processes**, then gathers the local outputs.

The report records timings over 30 runs per configuration. The current row partitions use local zero-padding without exchanging neighboring rows, so the timing comparison needs to be read alongside the [numerical-equivalence limitation](#benchmark-methodology-and-scope).

**Key files:** [Sequential implementation](Part%203/sequential.py) · [Parallel implementation](Part%203/parallel.py) · [MPI benchmark](Part%203/parallel_tb.py)

### 4. Distributed ResNet-50 with Horovod

Two implementations train **ResNet-50 from scratch on CIFAR-10**, resizing inputs to 224 × 224 pixels and running for four epochs. Both integrate Horovod for distributed optimization and initial model-state synchronization.

The PyTorch version uses `DistributedSampler`, sets the sampler epoch, and assigns GPUs by local rank. The TensorFlow version explicitly shards the data and builds a `tf.data` input pipeline with prefetching. The Slurm launcher requests **two nodes with two GPUs per node** and launches four MPI processes for PyTorch.

**Key files:** [PyTorch training](Part%204/pytorch_v.py) · [TensorFlow training](Part%204/tensorflow_v.py) · [Slurm launcher](Part%204/batch.sh)

*Part 4 contains source code and a launcher; no PDF report or benchmark results are included.*

### 5. OPT-125M training experiments with DeepSpeed

This project integrates **`facebook/opt-125m`**, Hugging Face `Trainer`, and DeepSpeed for causal language-model training on text from `yelp_review_full`. The experiments compare **one and two GPUs**, global batch sizes of **4 and 8**, and **Adam/ZeRO stage 3 versus AdamW/ZeRO stage 2**, with CPU offloading configured.

The committed script selects **four training examples and four evaluation examples**. These short runs exercise the training setup and expose overhead at a very small workload size. In the recorded experiments, adding a second GPU increased training runtime for every matched configuration.

**Key files:** [Training configuration](Part%205/LLM.py) · [Slurm launcher](Part%205/launch_slurm_llm.sh) · [Report and plots](Part%205/ML5.pdf)

## Recorded benchmark results

These are historical measurements from the included reports and logs. Their scope and limitations are documented below; the experiments have not been rerun for this README.

### MPI experiments

| Experiment | Sequential mean | Parallel mean | Configuration | Reported time ratio |
| --- | ---: | ---: | --- | ---: |
| [KNN prediction](Part%201/ML.pdf) | 799.5091 s | 14.5060 s | 64 MPI ranks | **55.13×** |
| [Convolution](Part%203/ML_Part_3.pdf) | 1.620217 s | 0.060880 s | 28 MPI ranks | **26.61×** |

Both reports summarize 30 measurements. The KNN timers cover different setup work, and convolution output equivalence is not established; these ratios describe the original experiments rather than a controlled, equivalent-workload speedup claim.

### OPT-125M training runtime

Three epochs on the four-example training subset. Values below come from the `train_runtime` entries in the saved logs and exclude dataset preparation and model loading before training.

| GPUs | Global batch size | Optimizer | ZeRO stage | Training runtime | Run log |
| ---: | ---: | --- | ---: | ---: | --- |
| 1 | 4 | Adam | 3 | 2.9806 s | [Log](Part%205/1_gpu_output4_adam_s3.txt) |
| 1 | 4 | AdamW | 2 | **1.9139 s** | [Log](Part%205/1_gpu_output4_adamW_s2.txt) |
| 1 | 8 | AdamW | 2 | 1.9221 s | [Log](Part%205/1_gpu_output8_adamW_s2.txt) |
| 2 | 4 | Adam | 3 | 3.2788 s | [Log](Part%205/2_gpu_output4_adam_s3.txt) |
| 2 | 4 | AdamW | 2 | 2.6813 s | [Log](Part%205/2_gpu_output4_atamW_s2.txt) |
| 2 | 8 | AdamW | 2 | 2.9823 s | [Log](Part%205/2_gpu_output8_atamW_s2.txt) |

The fastest recorded run used one GPU, global batch size 4, AdamW, and ZeRO stage 2. The small sample size limits conclusions about training throughput at scale; optimizer and ZeRO stage also change together in these comparisons.

## Running the projects

### Environment

Clone the repository and use a Python environment appropriate to the selected project:

```bash
git clone https://github.com/parsavares/ML-ALGO-HPC-Projects.git
cd ML-ALGO-HPC-Projects
```

| Part | Main Python dependencies | Execution requirements |
| --- | --- | --- |
| 1 | `numpy`, `mpi4py`; `tqdm` for the progress-bar variant | Compatible MPI runtime; memory for a training-set copy per rank |
| 2 | `jax`, `jaxlib`, `mpi4py` | Compatible MPI runtime |
| 3 | `numpy`, `jax`, `jaxlib`, `mpi4py`, `tensorflow`, `matplotlib` | MPI runtime; MNIST download/cache |
| 4 | `horovod` with the selected framework; `torch`, `torchvision`, `numpy`, `Pillow` or `tensorflow`, `numpy` | Framework-compatible MPI/CUDA environment; CIFAR-10 download/cache |
| 5 | `torch`, `transformers`, `datasets`, `accelerate`, `deepspeed` | Compatible GPU/CUDA environment; model and dataset download/cache |

The original scripts use university-specific modules and environment paths. There is no pinned dependency lockfile; adapt the software environment, Slurm directives, paths, and package-installation commands to your cluster before submission.

### Start with the ensemble example

With JAX, mpi4py, and a compatible MPI runtime installed, run this inside an allocation or on a machine with sufficient CPU resources:

```bash
cd "Part 2"
mpiexec -n 6 python ensemble_forecaster.py
```

The output lists the aggregate forecast statistics from rank 0.

### Slurm entry points

Run each command **from its corresponding project directory**, after adapting the environment settings.

| Part | Command | Workload |
| --- | --- | --- |
| 1 | `sbatch script.sh` | 30 independent sequential KNN runs launched concurrently |
| 1 | `sbatch scriptmpi.sh` | MPI KNN benchmark with 64 ranks |
| 3 | `sbatch batch_seq.sh` | Sequential convolution benchmark |
| 3 | `sbatch batch_parallel.sh` | Convolution benchmarks across five process counts |
| 4 | `sbatch batch.sh` | Four-process PyTorch/Horovod run; TensorFlow command is included but commented out |
| 5 | `sbatch launch_slurm_llm.sh` | One-GPU and two-GPU DeepSpeed runs on a single node |

Part 5 currently selects global batch size 8, AdamW, and ZeRO stage 2 in `LLM.py`. Recreating the other recorded configurations requires changing those settings and the output filenames. Its launcher retains a documented multi-node issue and currently requests one node.

## Benchmark methodology and scope

These academic experiments preserve the original implementations and reports. The following details matter when interpreting or extending them:

- **KNN:** the sequential timer includes data generation and fitting, while the MPI timer starts after setup and data distribution. Sequential repetitions run concurrently on 30 ranks. Queries are sampled from the training set, so prediction counts do not measure held-out generalization.
- **Convolution:** local chunks are zero-padded without halo exchange. A general kernel therefore sees different values at partition boundaries than in the full-image calculation. The reports use visual inspection, and the benchmark has no automated numerical-equivalence check or explicit JAX completion synchronization. Correct boundary handling and aligned timing are prerequisites for a rigorous scaling claim.
- **ResNet-50:** the code demonstrates distributed training mechanics. Evaluation metrics are computed per rank; global evaluation aggregation is not implemented. No accuracy or scaling result is claimed here.
- **OPT-125M:** four-example runs exercise the configuration, but do not establish model quality or large-dataset scaling. The PDF's loss labels differ from the saved training/evaluation logs; this README uses the logs' explicit runtime fields for its comparison.

## Authors and context

**Parsa Vares · Tom Walter · Luc Pereira Cardoso**

University of Luxembourg · *Programming Machine Learning Algorithms for HPC* · 2024

The reports describe collaborative work, including adaptations of provided course implementations. Project results are presented as team outcomes.

[Parsa Vares on GitHub](https://github.com/parsavares)
