# HGTs Repository

Welcome to the **HGTs** repository! This repository is dedicated to the development and exploration of advanced graph-based models, with a focus on Graph Neural Networks (GNNs) and Transformers. The `GraphLab` directory contains the core components of our models, including data loading, model creation, training, and utilities.

Steps in the pipeline:
1) Annotate cell types on the slides
2) Select non-overlapping regions with greatest diversity of cells - key cellular communities
3) In each key cellular community, construct cell graphs based on cell centroids using the Minimum Spanning Tree algorithm
4) Generate patient-level features
5) Transformer-based or graph-based methods were applied to model interactions between cellular communities, ultimately yielding a recurrence risk score. The recurrence risk score was then used to construct Kaplan-Meier curves for recurrencefree interval analysis, as well as to predict 1-year, 3-year, and 5-year recurrence outcome.

## Directory Structure

The `GraphLab` directory is organized into several key components:

- **model/**: This directory houses the core model implementations. It includes various GNNs, Transformer-based models, and other custom architectures.

- **utils/**: Utility scripts that assist in data processing, model training, and evaluation. This includes logging, checkpointing, and other helper functions.

## Key Scripts

- **DeepLoss.py**: Implements the deep loss functions used in training our models (not use).

- **DrawGraph.py**: Contains functions for visualizing graphs. This is useful for understanding the structure of the graph data and the model's learned features.

- **checkpoint.py**: Manages the saving and loading of model checkpoints during training. 

- **cmd_args.py**: Handles command-line arguments for configuring different aspects of the training and model setup. 

- **config.py**: Contains configuration settings, such as model parameters, paths, and other necessary settings.

- **loader.py**: Manages data loading and preprocessing.

- **logger.py**: Provides logging functionality to track training progress, metrics, and other important information.

- **loss.py**: Includes loss functions.

- **model_builder.py**: A utility script for constructing models based on the specified configurations. This helps in creating and initializing the model architecture.

- **optimizer.py**: Defines the optimization strategies used during training, such as SGD, Adam, etc.

- **register.py**: Manages the registration of different model components, ensuring that they can be easily referenced and utilized in the main training script.

- **train.py**: The main training script that integrates all the components and runs the model training process. This script ties together the data loading, model, optimizer, and loss functions.

The `Run` directory in this repository contains essential components and scripts required for executing various graph-based machine learning experiments. Below is a breakdown of the contents:

## Directory Structure

### `CreateGraph/`
This folder contains scripts or modules responsible for the creation and manipulation of graph data structures. It includes functionalities for:
- Generating graphs
- Defining nodes and edges
- Constructing specific types of graphs used in the model or experiments

### `configs/`
This directory stores configuration files (such as YAML or JSON) that define parameters and settings for different runs or experiments. The configuration files allow for:
- Easy modification of experiment parameters
- Replication of experiments with different settings without altering the codebase

### `dataloader.py`
This script is responsible for loading and preprocessing the data used in experiments. 


### `main.py`
The `main.py` file serves as the entry point for running the model or experiments. 
- Set up the experiment
- Initialize the model
- Execute the training and evaluation processes

## Dataset
We utilize the TCGA-LIHC dataset for our experiments. You can download the dataset from the following cloud storage address:
Cloud Storage Address: https://pan.baidu.com/s/12l-pDBlOxUtTK5EVQU4QVw?pwd=ve85 
Extract code：ve85 

## Getting Started

The commands below assume you work from the **`Run/`** directory (so `main.py` can import `GraphLab`). Paths in the code and YAML files still point to the authors’ machines; **replace them with your own** before running.

### Prerequisites

- **Python 3** with **PyTorch** (CUDA recommended).
- Core libraries used by the project include: **DGL**, **PyTorch Geometric** (via `GraphLab/loader.py`), **DeepSNAP**, **yacs**, **lifelines**, **scikit-learn**, **pandas**, **numpy**, **OpenSlide** (`openslide-python`), **OpenCV**, **Pillow**, **tqdm**, **histocartography**, **fuzzywuzzy**, **networkx**, **SciPy**. 
- See `environment_colab.yml` for requirements that were used for Colab run with CUDA 12.8 and pytorch 2.4

### 1. Data preparation

#### 1.1 Inputs you need per patient

The graph builder (`Run/CreateGraph/CreateMyGraph.py`) expects:

| Role | Content |
|------|--------|
| **Cell table** | `{label_data_path}/{PatientID}/{PatientID}.txt` — tab-separated export of cell morphology / positions (see script for column usage). |
| **Label images** | `{PatientID}-CellLabels.png`, `{PatientID}-NucleiLabels.png` under the same patient folder. |
| **WSI** | `{WSI_data_path}/{PatientID}/{PatientID}.{ndpi,svs}` (OpenSlide). |
| **Follow-up / outcomes** | A tab-separated file (see `--follow_up_data` in `Run/CreateGraph/options.py`) whose **column names must match** what the code indexes (e.g. specimen ID, recurrence, survival columns as used in `CreateMyGraph.py`). |

Cell types in the TXT are fuzzy-matched to the `CellTypes` list in `CreateMyGraph.py`.

#### 1.2 Build cell graphs (MST patches)

1. Edit **`Run/CreateGraph/options.py`** defaults (`--label_data_path`, `--WSI_data_path`, `--follow_up_data`) or pass CLI flags.
2. Edit the **`if __name__ == '__main__'`** block in **`CreateMyGraph.py`**: it currently hardcodes output roots and patient lists (e.g. `result_dataset`, `Patients = os.listdir(...)`). Point these to **your** output directory and cohort.
3. Run from `Run/CreateGraph/`:

   ```bash
   cd CreateGraph
   python CreateMyGraph.py
   ```

For each selected patch, the script writes a folder containing at least **`AllCell.bin`** (DGL graph via `dgl.data.utils.save_graphs`). Optional visualization writes `wsi.png` / `node_in_img.png` when those paths exist.

#### 1.3 Train / validation / test layout

Training with `dataset.format: dglmulty` expects this structure (see `load_dgl_Multy` in `GraphLab/loader.py`):

```text
{dataset.dir}/
  {dataset.name}/
    train/
      {PatientID}/
        {patch_box_name}/     # one folder per patch
          AllCell.bin
    val/
      {PatientID}/...
    test/
      {PatientID}/...
```

- **`Run/CreateGraph/Split_pro.py`** is an example that splits patient folders into `train` / `val` / `test` under a chosen key (e.g. `TCGA`) using a censoring CSV. Adjust **`cg_path`**, **`censor_file`**, and **`sava_path`** to match your machine and naming.
- Alternatively, copy or symlink patient directories into the three splits yourself.

#### 1.4 Risk / covariate CSV (training)

`load_dgl_Multy` **loads a separate CSV** of normalized risk scores and merges it with graph labels. The path is **hardcoded** in `GraphLab/loader.py` (search for `risk_csv = pd.read_csv`). **Change that path** to your file and ensure column names (`标本号`, `风险系数`, etc.) match your tables or refactor the column names in code to match an English CSV.

---

### 2. Configure training

1. Copy **`Run/configs/IDGNN/graph.yaml`** and edit at minimum:
   - **`out_dir`**: base directory for logs and checkpoints (actual run dir becomes `{out_dir}/{yaml_stem}/` — see `set_out_dir` in `GraphLab/config.py`).
   - **`dataset.dir`**: parent folder that contains **`{dataset.name}/train|val|test`** (see §1.3).
   - **`dataset.name`**: subdirectory name under `dataset.dir` (must match your on-disk folder).
   - **`dataset.format`**: use **`dglmulty`** for multi-patch-per-patient graphs (default in the sample YAML).
   - **`dataset.task`**, **`dataset.task_type`**, **`model.loss_fun`**: e.g. graph-level regression with **`cox`** loss for survival-style labels (as in the sample).
   - **`train` / `model` / `gnn` / `optim`**: batch size, epochs, architecture (`layer_type`, `layers_mp`, pooling, Transformer options, etc.).

2. Optional **overrides** on the command line use YACS style after the config path, e.g.  
   `python main.py --cfg configs/IDGNN/graph.yaml train.batch_size 16`

---

### 3. Run training

From **`Run/`**:

```bash
cd /path/to/liver_cell_gnn/Run
python main.py --cfg configs/IDGNN/graph.yaml
```

- **`--repeat N`**: repeat the experiment with different seeds (see `main.py`).
- **`--mark_done`**: renames the YAML to `*_done` after a successful run (optional batch convenience).

Checkpoints and configs are written under **`cfg.run_dir`** (per seed, under `cfg.out_dir`). See `GraphLab/checkpoint.py` and `GraphLab/logger.py` for what is saved.

**Resume / pretrained weights:** set **`train.auto_resume`** and optionally **`train.resume_path`** in the YAML to continue from or initialize from a checkpoint (paths in the repo are examples only).

---

### 4. Evaluation and inference notes

- **Validation and test** are run inside the training loop on each eval epoch (`GraphLab/train.py`: `eval_epoch` on loaders after the train loader). Metrics and losses are logged via the logger; the exact survival metrics depend on `task_type` and `loss_fun`.
- For **`task_type: classification2regression`**, the code aggregates predictions and fits a **Cox PH model** (`lifelines.CoxPHFitter`) and prints **concordance index** on the test split (see `train()` in `GraphLab/train.py`). The default sample **`graph.yaml`** uses **`regression`** with **`cox`** loss instead; behavior follows `compute_loss` in `GraphLab/loss.py`.
- There is **no separate inference-only script** in the repo. To “run inference” on new patients, prepare their graphs in the same **`AllCell.bin`** layout, add them under a split (or extend the loader), point **`dataset.dir`** / **`dataset.name`** at that tree, and either run training with **`optim.max_epoch`** minimal and eval-only logic, or add a small script that loads **`create_model`** + **`load_ckpt`** and runs **`eval_epoch`** — you would need to implement that glue code locally.

---

### 5. Reproducibility checklist

1. Replace all **absolute paths** in YAML, `options.py`, `CreateMyGraph.py`, `Split_pro.py`, and **`GraphLab/loader.py`** (dataset dir, risk CSV, optional cache paths).
2. Align **follow-up and risk tables** with the column names expected in code.
3. Confirm **DGL** can load each **`AllCell.bin`** and that every patch folder contains the expected files.
4. Set **`cfg.seed`** (and `--repeat`) for repeatable runs; use **`agg_runs`** output under `out_dir` when using multiple seeds.

For **TCGA-LIHC** raw data, see the **Dataset** section above; preprocessing into the folder layout in §1 is project-specific.

