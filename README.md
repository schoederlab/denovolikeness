# Denovolikeness
Code for the paper "Exploration of denovo protein properties and similar natural proteins"


## Installation

### Prerequisites
- Conda or Miniconda
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/JohannesKlier/denovolikeness
cd denovolikeness
```

2. Set up the conda environment:


```bash
chmod +x setup.sh
./setup.sh
```

This will create a conda environment named `denovolikeness` with all necessary dependencies including:
- Python 3.11
- TensorFlow 2.16.1
- PyTorch
- PyRosetta
- Scientific computing libraries (numpy, pandas, scikit-learn, matplotlib, seaborn, scipy)
- Jupyter for notebook analysis



### Verification

To verify the installation:
```bash
python -c "import tensorflow as tf; import pyrosetta; print('Installation successful!')"
```

## Classifying Proteins

To run predictions on a directory of PDB files:

```bash
cd predict
python predict.py <pdb_directory>
```

This will:
1. Process each PDB file in the directory
2. Calculate structural features
3. Generate predictions based on features
4. Save results to CSV files:
   - `<directory_name>_features.csv`: Contains calculated features
   - `<directory_name>_predictions.csv`: Contains final predictions



### Command-line options

The prediction script `predict.py` accepts the following command-line arguments:

-   `pdb_directory`: (Required) The path to the directory containing the PDB files you want to analyze.
-   `--no-logging`: (Optional) Disable logging to the console.
-   `--save-structure`: (Optional) Save the PDB files of the first chain of each processed protein.
-   `--no-overwrite`: (Optional) Prevent the script from overwriting existing prediction and feature files.

## Project Structure

The project is organized into several main components:

### 1. Prediction (`/predict`)
Model for classifying proteins based on their structural features.

### 2. Training (`/training`)
Code and models for training the classification model:
- `src/`: Training source code (training.py, util.py, visualisation.py)
- `models/`: Trained model files and training histories
- `scaler.pkl`: Pre-trained feature scaler
- `metrics_results.csv`: Training metrics and results

### 3. Data (`/data`)
Contains training data, feature calculations, and prediction results:
- `training_data.csv`: Training dataset
- `features_screen/`: Calculated features for all screened organisms
- `predictions_screen/`: Denovolike scores for all screened organisms
- `sequence_info/`: Sequence information and metadata for all screened organisms (download via zenodo)

### 4. Figures (`/figures`)
Analysis notebooks and visualization code.


### 5. Supplements (`/supplements`)
Additional analysis and experimental work.






## Citation

If you use this software in your research, please cite:
```
Klier, J., et al. (2026).
```


## Contact

klier@izbi.uni-leipzig.de 