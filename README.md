# DKTD

Core implementation of **DKTD: A Dynamic Knowledge Transfer Decision-Based Clustered Multi-Teacher Distillation Approach for Federated Learning**.

**Authors:** Wei Ma, Angran Xia, and Jie Dong.

DKTD combines client clustering, intra-cluster federated training, and target-aware cross-cluster knowledge distillation. Candidate teacher clusters are ranked using class complementarity, predictive confidence, and prototype-based transfer risk. Selected teachers provide a score-weighted prediction ensemble for client-local distillation.

This release provides two self-contained entry points:

| Script | Supported datasets | Purpose |
|---|---|---|
| `fedcmt-dktd.py` | CIFAR-10 | Main training runner, teacher-strategy comparisons, and global-test/cluster-holdout evaluation |
| `dktd_incluster_eval_multidataset.py` | CIFAR-10, Fashion-MNIST, MNIST | DKTD training and fixed-cluster pre/post-distillation holdout evaluation |

## Method overview

1. **Client clustering.** Extract label distributions, class prototypes, and prototype histograms from client training data. Construct a combined distance matrix and partition clients with K-Medoids.
2. **Intra-cluster training.** Train clients locally and aggregate their model updates within each cluster using sample-weighted federated averaging.
3. **Target-aware distillation.** Evaluate candidate teachers for each target cluster, select the top-M teachers, and update client-local students using cross-entropy and knowledge distillation. Aggregate the updated students within their target cluster.

Cluster assignments and initial summaries remain fixed during training. Teacher confidence, rankings, and fusion weights are reassessed at scheduled distillation cycles.

## Requirements

The implementation uses Python and the following packages:

- PyTorch (`torch`)
- torchvision
- NumPy
- pandas
- Matplotlib

Install the dependencies in your Python environment:

```bash
python -m pip install torch torchvision numpy pandas matplotlib
```

Use a compatible PyTorch/torchvision installation for your operating system and CUDA environment. Both scripts support CUDA and CPU execution; `--device auto` selects CUDA when available.

Both entry points are self-contained and do not require additional project-specific Python modules.

## Dataset and model

| Dataset | Model | Entry point |
|---|---|---|
| CIFAR-10 | torchvision ResNet-18 with a ten-class classifier | Both scripts |
| Fashion-MNIST | Two-convolution CNN for single-channel 28 x 28 images | `dktd_incluster_eval_multidataset.py` |
| MNIST | Two-convolution CNN for single-channel 28 x 28 images | `dktd_incluster_eval_multidataset.py` |

- **Client partitioning:** class-wise Dirichlet partitioning controlled by `--beta`.
- **Client data split:** nominally 10% validation, 10% cluster-specific holdout, and the remaining examples for training. Integer allocation may change the exact fractions for small class subsets.
- **Teacher confidence:** mean maximum softmax probability on the teacher cluster's validation examples.
- **Evaluation:** the main runner uses the official CIFAR-10 test set and cluster-specific holdout data. The multi-dataset runner evaluates each cluster on its fixed holdout data before and after scheduled distillation updates.

The default data directory is `./data`. The main runner downloads CIFAR-10 automatically when needed. For the multi-dataset runner, add `--download` to download the selected dataset; omit it when the dataset is already available in `--data-root`.

For CIFAR-10, ImageNet-pretrained ResNet-18 weights are requested by default. Use `--no-pretrained` for random initialization. If pretrained loading fails, the script reports the fallback and records its initialization status in `run_config.json` under `pretrained_load_status`. Fashion-MNIST and MNIST use randomly initialized dataset-specific CNNs; `--pretrained` applies only to CIFAR-10.

## Run CIFAR-10 with the main runner

Run the commands below from the directory containing `fedcmt-dktd.py`.

Display the command-line options:

```bash
python fedcmt-dktd.py --help
```

Run DKTD with 50 clients, five clusters, three selected teachers, and 100 communication rounds:

```bash
python fedcmt-dktd.py --methods full_dktd --beta 0.2 --seed 0 --clusters 5 --teachers 3 --rounds 100 --workers 0 --device auto --data-root ./data --result-root ./results/dktd
```

For the moderate label-skew setting, change `--beta 0.2` to `--beta 0.5`:

```bash
python fedcmt-dktd.py --methods full_dktd --beta 0.5 --seed 0 --clusters 5 --teachers 3 --rounds 100 --workers 0 --device auto --data-root ./data --result-root ./results/dktd
```

Use `--device cpu` to select CPU execution explicitly, or `--device cuda` to require CUDA. The examples use `--workers 0` for a straightforward single-process data-loading setup.

Specify `--methods full_dktd` to run DKTD alone. Omitting `--methods` launches the default group of comparison methods.

The main runner accepts one `--seed` per invocation. Its dataset option supports CIFAR-10; use the multi-dataset runner below for Fashion-MNIST and MNIST.

### Ten-seed CIFAR-10 training

Ten seeds means ten independent training runs with seeds 0 through 9. Each run below uses **100 communication rounds**.

Windows PowerShell:

```powershell
foreach ($seed in 0..9) {
    python fedcmt-dktd.py --methods full_dktd --beta 0.2 --seed $seed --clusters 5 --teachers 3 --rounds 100 --workers 0 --device auto --data-root ./data --result-root ./results/dktd_10seeds_beta0.2
    if ($LASTEXITCODE -ne 0) { throw "DKTD failed for seed $seed" }
}
```

Linux/macOS Bash:

```bash
for seed in {0..9}; do
    python fedcmt-dktd.py --methods full_dktd --beta 0.2 --seed "$seed" --clusters 5 --teachers 3 --rounds 100 --workers 0 --device auto --data-root ./data --result-root ./results/dktd_10seeds_beta0.2 || exit 1
done
```

After the runs complete, summarize the saved results:

```bash
python fedcmt-dktd.py --aggregate --aggregate-group main --result-root ./results/dktd_10seeds_beta0.2
```

The summary files are written under `./results/dktd_10seeds_beta0.2/_aggregates/main/`. Use one result root for a consistent training configuration and code version. For beta = 0.5, change both `--beta 0.2` and the result-directory name.

## Run three datasets with ten seeds

The multi-dataset runner accepts `--seeds 0-9` directly and runs the ten seeds sequentially. It trains DKTD and records cluster-specific holdout measurements around scheduled distillation updates. Each command below uses 50 clients, K = 5, M = 3, and 100 communication rounds per seed.

Keep `dktd_incluster_eval_multidataset.py` alongside this README to use these commands.

### Fashion-MNIST

```bash
python dktd_incluster_eval_multidataset.py --dataset fashion_mnist --beta 0.2 --seeds 0-9 --K 5 --M 3 --clients 50 --rounds 100 --workers 0 --device auto --download --data-root ./data --result-root ./results/incluster/fashion_mnist_beta0.2
```

### MNIST

```bash
python dktd_incluster_eval_multidataset.py --dataset mnist --beta 0.2 --seeds 0-9 --K 5 --M 3 --clients 50 --rounds 100 --workers 0 --device auto --download --data-root ./data --result-root ./results/incluster/mnist_beta0.2
```

### CIFAR-10

```bash
python dktd_incluster_eval_multidataset.py --dataset cifar10 --beta 0.2 --seeds 0-9 --K 5 --M 3 --clients 50 --rounds 100 --workers 0 --device auto --download --data-root ./data --result-root ./results/incluster/cifar10_beta0.2
```

To run the beta = 0.5 setting, replace `--beta 0.2` with `--beta 0.5` and use a result root ending in `beta0.5`. The dataset names `fashion-mnist` and `FashionMNIST` are also accepted for Fashion-MNIST.

The runner summarizes saved runs after the requested seeds finish and prints the path to `incluster_paired_table.csv`. To regenerate the summary without training, point `--aggregate` at the same result root, for example:

```bash
python dktd_incluster_eval_multidataset.py --aggregate --result-root ./results/incluster/mnist_beta0.2
```

Summary files are written under `<result-root>/original_cluster_multidataset_summary/`, including `retention_per_seed.csv`, `retention_across_seed_summary.csv`, and `incluster_paired_table.csv`.

## Main runner settings

The following settings apply to `fedcmt-dktd.py`:

| Setting | Default value | Configuration |
|---|---|---|
| Number of clients | 50 | `--clients` |
| Communication rounds | 100 | `--rounds` |
| Number of clusters K | 5 | `--clusters` or `--K` |
| Selected teachers M | 3 | `--teachers` or `--M` |
| Dirichlet parameter beta | 0.2 | `--beta` |
| Random seed | 2 | `--seed` |
| Local training epochs | 1 | `--local-epochs` |
| Batch size | 32 | `--batch-size` |
| Distillation interval | Every 5 rounds | `--distill-interval` |
| Distillation epochs per cycle | 1 | `--distill-epochs` |
| Local learning rate | 0.01 | `LR` |
| Distillation learning rate | 0.01 | `DISTILL_LR` |
| Distillation temperature | 4.0 | `DISTILL_TAU` |
| KD loss coefficient | 0.7 | `KD_LAMBDA` |
| Class-demand coefficient | 5.0 | `DKTD_RHO` |
| Complementarity/confidence/risk coefficients | 1.0 / 0.5 / 0.8 | `DKTD_ALPHA`, `DKTD_BETA_D`, `DKTD_GAMMA` |

Names beginning with `--` are command-line options. Uppercase names are constants near the top of `fedcmt-dktd.py`. The cluster and teacher counts must satisfy `2 <= K <= clients` and `1 <= M < K`.

The multi-dataset runner exposes its own options, including `--lr`, `--distill-lr`, `--tau`, and `--kd-lambda`. Display its full option list with:

```bash
python dktd_incluster_eval_multidataset.py --help
```

## Teacher strategies

The main script also includes the following teacher-strategy options:

| Method identifier | Description |
|---|---|
| `full_dktd` | Target-aware teacher selection with score-weighted fusion |
| `dktd_topm_equal` | The same teacher-selection rule with equal fusion weights |
| `all_equal` | All active candidate teachers with equal weights |
| `random_m` | A random subset of candidate teachers with equal weights |
| `similarity_topm` | Similarity-based teacher selection |
| `dktd_no_cross_distill` | Intra-cluster training without cross-cluster distillation |

Select multiple methods with a comma-separated list:

```bash
python fedcmt-dktd.py --methods full_dktd,dktd_topm_equal,all_equal,random_m --beta 0.2 --seed 0 --workers 0 --result-root ./results/teacher_strategies
```

## Main runner output files

For the first DKTD example, the run directory is organized as follows:

```text
results/dktd/
└── main/
    └── cifar10_K5_M3_beta0.2_seed0/
        ├── run_config.json
        ├── client_splits.json
        ├── client_cluster_assignment.csv
        ├── cluster_summary.csv
        ├── cluster_analysis.png
        ├── summary.csv
        └── full_dktd/
            ├── round_metrics.csv
            ├── cluster_metrics.csv
            ├── distill_events.csv
            ├── distill_candidates.csv
            ├── resource_metrics.csv
            ├── summary.json
            └── accuracy_curve.png
```

- `run_config.json` records the configuration, initialization status, package versions, and run status.
- `client_splits.json` and `client_cluster_assignment.csv` record the client data partitions and cluster assignments.
- `round_metrics.csv` and `cluster_metrics.csv` contain round-level and cluster-level evaluation results.
- `distill_candidates.csv` records teacher-score components, selection decisions, and fusion weights.
- `distill_events.csv` records measurements immediately before and after target-cluster distillation updates.
- `resource_metrics.csv` contains timing, update counts, and estimated model-payload volumes.
- `summary.csv` and the method-specific `summary.json` provide run summaries.

DKTD also writes original-cluster evaluation reports in its method directory. Global-test metrics and own-cluster holdout metrics describe different evaluation distributions. Pre/post-distillation measurements describe individual updates within a training run.

If a nonempty run directory already exists, the script creates a new directory with a repeat suffix. Output files are CSV, JSON, and PNG reports; this entry point does not save model-weight checkpoints.
