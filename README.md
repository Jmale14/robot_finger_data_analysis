# Robot Finger Data Analysis

## Overview
Dataset and methods used for predicting material texture and softness from a tactile robotic finger.

Data collected in 3 separate trial type.
- Texture only: 18 different material texture.
- Softness only: 4 different softness levels.
- Texture & Softness: 5 different textures and 4 different softness levels.

For each matieral type three trials are conducted. Each trial comprises the robotics finger sliding over the material 10 times.


## Project Structure
```
├── raw_data/                          # Raw data files for each of the 3 trial types.
├── processed_data/                    # Folded, normalized, and optionally PCA-transformed data.
├── results/                           # Training runs, confusion matrices, and analysis outputs.
├── utils/                             # Utility functions for data prep, training, and plotting.
├── config/                            # Configuration files, including hyperparameters and environment setup.
├── README.md                          # This file
```

## Runnable Scripts
The scripts below are intended to be run from the repository root.

- analyse_pca_components.py: Inspect fitted PCA models, summarize component loadings, and save PCA-related plots to results/pca_analysis.
- analyse_uni_vs_multi_factor_recog.py: Evaluate trained uni-modal versus multi-factor recognition models and produce metrics/confusion-matrix summaries.
- prepare_data_folds.py: Build sliding-window samples from the raw data, split them into folds, normalize them, optionally apply PCA, and save the processed data under processed_data/.
- run_training.py: Train and evaluate a model for a chosen dataset, recognition target, modality, and PCA setting. Results and plots are written to results/ and trained models can be saved to models/.
- run_multiple_trials.sh: Convenience wrapper for launching many training runs in sequence using either a JSON config or explicit CLI argument lists.

## Link to Publication
TBD

## Running Multiple Trials

- Use the `run_multiple_trials.sh` wrapper to run `run_training.py` for one or more trials sequentially.
- The wrapper supports two modes:
    - JSON config: provide `--config config/trials_example.json` (parses an array of trial objects). Requires `jq` to be installed.
    - Comma-separated lists: provide `--datasets`, `--recognition-types`, `--modalities`, `--models`, and `--use-pca-options` to generate a Cartesian product of trials.

- Example (JSON config):

```bash
bash run_multiple_trials.sh --config config/trials_example.json --delay 2 --log results/trials_log.txt
```

- Example (lists):

```bash
bash run_multiple_trials.sh --datasets text&soft --recognition-types softness --modalities all --models CNN-LSTM,CNN --use-pca-options true,false --delay 2 --log results/trials_log.txt
```

- Notes:
    - PCA (`--use-pca`) is only applied when `--modality all` is used; the wrapper disables PCA for single-modalities automatically.
    - The wrapper will try to activate `.venv` in the repository root if present.
    - To forward arbitrary single-run arguments to `run_training.py`, simply pass them and they will be forwarded (e.g. `--dataset texture --modality press`).
        - You can append a custom subfolder to the results `save_folder` using `--save-folder-app "my_run"`.
            This will cause results to be saved under the generated save folder plus the appendix (e.g. `results/text&soft_softness_pcaTrue_CNN-LSTM/my_run`).
            For JSON configs, include `"save_folder_app": "my_run"` in a trial object to set it per-trial.

See `config/trials_example.json` for a sample configuration file.