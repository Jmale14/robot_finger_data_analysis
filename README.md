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
├── raw_data/               # Raw data files for each of the 3 trial types.
├── utils/                  # Utility functions
├── hparams.py/             # Hyperparameters used for each classification task.
├── prepare_data_folds.py/  # Convert raw data into normalisaed windows ready for classification. Saves to /preprocessed_data.
├── run_training.py/        # Train and evaluate CNN-LSTM model on selected dataset type. Output results to /results and saved models to /models.
├── README.md               # This file
```

## Link to Publication
    #####

## Running Multiple Trials

- Use the `run_multiple_trials.sh` wrapper to run `run_training.py` for one or more trials sequentially.
- The wrapper supports two modes:
    - JSON config: provide `--config trials_example.json` (parses an array of trial objects). Requires `jq` to be installed.
    - Comma-separated lists: provide `--datasets`, `--recognition-types`, `--modalities`, `--models`, and `--use-pca-options` to generate a Cartesian product of trials.

- Example (JSON config):

```bash
bash run_multiple_trials.sh --config trials_example.json --delay 2 --log results/trials_log.txt
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

See `trials_example.json` for a sample configuration file.