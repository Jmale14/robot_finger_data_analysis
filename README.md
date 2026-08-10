# Robot Finger Data Analysis

## Overview

Dataset and methods used for predicting material texture and softness from a tactile robotic finger.

Data were collected in 3 separate trial types:

- Texture only: 18 different material textures.
- Softness only: 4 different softness levels.
- Texture & Softness: 5 different textures and 4 different softness levels.

For each material type, three trials were conducted. Each trial comprises the robotic finger sliding over the material 10 times.

## Repository Structure

```
├── raw_data/                          # Raw sensor recordings.
├── processed_data/                    # Normalized sliding windows and optional PCA-transformed data.
├── processed_data_3fold/              # Alternative 3-fold processed dataset layout.
├── results/                           # Training outputs, confusion matrices, and analysis results.
├── models/                            # Saved trained model files.
├── utils/                             # Utility functions for preprocessing, plotting, and training.
├── config/                            # Experiment configuration, JSON examples, and hyperparameters.
├── README.md                          # This file.
├── requirements.txt                   # Python dependencies for reproducible execution.
```

## Prerequisites

- Python 3.11+ (compatible with TensorFlow 2.20).
- Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

- If you are using the shell wrapper, a POSIX-compatible shell is required for `run_multiple_trials.sh`.

## Runnable Scripts

The scripts below are intended to be executed from the repository root.

- `prepare_data_folds.py`

  - Builds sliding-window samples from raw sensor files.
  - Splits data into folds, normalizes it, optionally applies PCA, and saves processed fold files under `processed_data/`.
- `run_training.py`

  - Trains and evaluates a model for a chosen dataset, recognition target, modality, PCA flag, and architecture.
  - Writes metrics, plots, and optional saved models to `results/` and `models/`.
- `run_multiple_trials.sh`

  - Wrapper to launch multiple `run_training.py` experiments sequentially from a JSON config or comma-separated option lists.
- `analyse_pca_components.py`

  - Inspects fitted PCA models, summarizes component loadings, and saves PCA analysis plots to `results/pca_analysis/`.
- `analyse_feat_permutation.py`

  - Runs feature ablation and permutation importance studies for trained CNN-LSTM models.
  - Produces results under `results/feat_importance_analysis/`.
- `plot_feature_permutation_results.py`

  - Aggregates and plots summary metrics from the feature importance analysis output.

## Running Training Experiments

Example single training run:

```bash
python run_training.py --dataset "text&soft" --recognition-type softness --modality all --model-type CNN-LSTM --use-pca --folds 5
```

- `--dataset` supports `text&soft`, `texture`, or `softness`.
- `--recognition-type` chooses the prediction target and is required for combined datasets.
- `--modality` supports `all`, `accel`, `gyro`, `press`, or `feat_study`.
- `--use-pca` is only used for `modality all` or `modality feat_study`.
- `--output-model` saves the trained model in the `models/` folder.

## Running Multiple Trials

Use `run_multiple_trials.sh` to execute `run_training.py` for one or more trials sequentially many .

- JSON configuration example:

```bash
bash run_multiple_trials.sh --config config/trials_example.json --delay 2 --log results/trials_log.txt
```

- Comma-separated list example:

```bash
bash run_multiple_trials.sh --datasets "text&soft" --recognition-types softness --modalities all --models CNN-LSTM,CNN --use-pca-options true,false --delay 2 --log results/trials_log.txt
```

- Notes:
  - `run_multiple_trials.sh` uses the local `.venv` activation if the virtual environment directory exists.
  - When `--modality` is not `all` or `feat_study`, the wrapper forces `--use-pca` off because PCA only applies to combined-time modalities.
  - Add `--save-folder-app "my_run"` to create a custom subfolder suffix for outputs.

## Feature Importance Analysis

This repository includes a pipeline for feature ablation and permutation importance analysis.

- Run the main feature importance script:

```bash
python analyse_feat_permutation.py --feat_to_blank permutation
```

- Common options:

  - `--recognition-type [texture|softness]` — target label type.
  - `--feat_to_blank [accel|gyro|press|permutation|None]` — choose the feature group to blank or run permutation importance.
  - `--text-soft` — use the unimodal dataset instead of the combined `text&soft` dataset.
  - `--all-variants` — run both recognition targets and both combined/unimodal variants.
  - `--no-plot` — disable plotting of confusion matrices.
  - `--save-folder-app <suffix>` — append a custom suffix to the output folder.
- Default mode runs a single specified variant. Add `--all-variants` to execute all supported feature importance combinations.
- After running feature importance, use `plot_feature_permutation_results.py` to generate summary charts from the saved trial outputs:

```bash
python plot_feature_permutation_results.py
```

## Results Organization

- `results/pca_analysis/` — PCA component and variance analysis.
- `results/feat_importance_analysis/` — ablation and permutation importance study outputs.
- `results/<dataset>_pca<True|False>_<model>/` — main training experiment output folders.

## Link to Publication

TBD

## Notes

- `requirements.txt` lists the Python dependencies required for reproducible execution.
- The repository is designed to support both model training and analysis workflows for publication replication and review.
