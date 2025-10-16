# Denovolikeness
A tool for assessing the de novo likeness of protein structures through structural scoring and classification.

## Project Structure

The project is organized into several main components:

### 1. Prediction (`/predict`)
Machine learning model for classifying proteins based on their structural scores.

### 2. Training (`/training`)
Code for training the classification model.

### 3. Data (`/data`)
Contains training data, feature calculations, and prediction results:
- `training_data.csv`: Training dataset
- `features_screen/`: Calculated features for different organisms
- `predictions_screen/`: Prediction results for screened organisms
- `sequence_info/`: Organism sequence information and metadata

### 4. Figures (`/figures`)
Analysis notebooks and visualization code:
- `screening/`: Cross-organism screening analysis and plots
- `features/`: Feature analysis and visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JohannesKlier/denovolikeness
cd denovolikeness
```

2. Set up the conda environment using the provided script:
```bash
chmod +x setup.sh
./setup.sh
```

This will create a conda environment with all necessary dependencies including:
- Python 3.11
- TensorFlow 2.16.1
- PyTorch
- PyRosetta
- Scientific computing libraries (numpy, pandas, scikit-learn, matplotlib)

3. Activate the environment:
```bash
conda activate denovolikeness
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
3. Generate predictions
4. Save results to CSV files:
   - `<directory_name>_features.csv`: Contains calculated structural features
   - `<directory_name>_predictions.csv`: Contains final predictions

### Command-line options

The prediction script `predict.py` accepts the following command-line arguments:

-   `pdb_directory`: (Required) The path to the directory containing the PDB files you want to analyze.
-   `--no-logging`: (Optional) Disable logging to the console.
-   `--save-structure`: (Optional) Save the PDB files of the first chain of each processed protein.
-   `--no-overwrite`: (Optional) Prevent the script from overwriting existing prediction and feature files.

### Output Files

1. Features File (`<directory_name>_features.csv`):
   - Contains structural features for each protein
   - Includes: -logP_Struc, -logP_Contact, area_sc, per_con, per_no_con, contact_order

2. Predictions File (`<directory_name>_predictions.csv`):
   - Contains final predictions for each protein
   - Format: `UniProtID,scores`

## Analysis and Visualization

The `/figures` directory contains Jupyter notebooks for comprehensive analysis:

### Cross-Organism Screening (`/figures/screening/`)
- `screening.ipynb`: Complete analysis pipeline including:
  - Feature calculation across multiple organisms
  - Model predictions and scoring
  - Phylogenetic analysis with dendrograms
  - Statistical visualization (bar plots, violin plots, swarm plots)
  - Cross-species comparison of denovolikeness

### Feature Analysis (`/figures/features/`)
- Analysis of structural features used in the classification model


## Citation

If you use this software in your research, please cite:
```
Klier, J., et al. (2025).
```


## Contact

klier@izbi.uni-leipzig.de 