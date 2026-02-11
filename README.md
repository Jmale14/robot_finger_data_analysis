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